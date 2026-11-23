"""Independent sealed readout replay; imports no learner or evaluator module.

The complete fit and evaluation seals are required before any data array is
opened. All point metrics are reconstructed. Three prespecified literal
bootstrap covariances check every scoring context; all 200 nonzero-prediction
error identities are checked through the matched low-rank source projections.
The latter checks are conditional on the saved zero-error norm, independently
checked in those literal resamples and by the separate Gram algebra audit.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
LITERAL_RESAMPLES = (0, 99, 199)
MAX_DIFFERENCE = 0.0


def read_json(path):
    return json.loads(Path(path).read_text())


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, rtol=2e-8, atol=2e-10):
    global MAX_DIFFERENCE
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    good = np.isfinite(actual) & np.isfinite(expected)
    if good.any():
        MAX_DIFFERENCE = max(MAX_DIFFERENCE, float(np.max(np.abs(actual[good] - expected[good]))))


def cov(left, right=None):
    """Use NumPy's concatenated covariance as a separate implementation."""
    left = np.asarray(left, float)
    if right is None:
        return np.cov(left, rowvar=False, bias=True)
    right = np.asarray(right, float)
    p = left.shape[1]
    return np.cov(np.column_stack([left, right]), rowvar=False, bias=True)[:p, p:]


def errors(record, prediction, truth, mask=None):
    mask = np.ones(truth.shape, bool) if mask is None else mask
    assert record['entries'] == int(mask.sum())
    if not mask.any() or not np.isfinite(prediction[mask]).all() or not np.isfinite(truth[mask]).all():
        assert record['status'] == 'unscorable' and record['rmse'] is None and record['mae'] is None
        return
    assert record['status'] == 'ok'
    difference = prediction[mask] - truth[mask]
    close(record['rmse'], np.linalg.norm(difference) / np.sqrt(difference.size))
    close(record['mae'], np.abs(difference).mean())


def summary_correlations(record, matrix):
    assert record['score_count'] == matrix.shape[0] and record['state_count'] == matrix.shape[1]
    assert record['valid_entries'] == int(np.isfinite(matrix).sum())
    for index, column in enumerate(matrix.T):
        finite = np.isfinite(column)
        saved = record['max_absolute_correlation_per_state'][index]
        if not finite.any():
            assert saved is None
        else:
            close(saved, np.max(np.abs(column[finite])))


def boot_inner_from_sources(left, right, left_loading, right_loading, signs, weights):
    """<Cov_boot(left,right), sum sign*b_left*b_right'> using only k projections."""
    if len(signs) == 0:
        return np.zeros(len(weights))
    left_score = left @ left_loading
    right_score = right @ right_loading
    n = len(left)
    cross = weights @ (left_score * right_score) / n
    mean_left = weights @ left_score / n
    mean_right = weights @ right_score / n
    return (cross - mean_left * mean_right) @ signs


def require_seals():
    """Do not read scientific outcomes or data without both completed blocks."""
    fit_path, eval_path = STUDY / 'runs_v1', STUDY / 'evaluation_v1'
    fit_seal = read_json(fit_path / 'provenance.json')
    eval_seal = read_json(eval_path / 'provenance.json')
    assert fit_seal['evaluator_read'] is False and eval_seal['status'] == 'complete'
    assert eval_seal['training_or_selection_performed'] is False
    protocol = read_json(STUDY / 'protocol.json')
    assert protocol['expected_arm_records'] == 225
    for name, digest in fit_seal['source_sha256'].items():
        assert sha(STUDY / name) == digest, name
    for name, digest in fit_seal['support_sha256'].items():
        assert sha(STUDY.parent / name) == digest, name
    for folder, seal in [(fit_path, fit_seal), (eval_path, eval_seal)]:
        assert 'summary.json' in seal['artifacts']
        for name, digest in seal['artifacts'].items():
            assert sha(folder / name) == digest, (str(folder), name)
    evaluation_sources = {'evaluator': STUDY / 'evaluate_extensions.py',
                          'utilities': STUDY.parent / 'count_mdcrl_2026_11_01/evaluate_count_sources.py',
                          'protocol': STUDY / 'protocol.json',
                          'fit_seal': fit_path / 'provenance.json'}
    for key, path in evaluation_sources.items():
        assert sha(path) == eval_seal['source_sha256'][key]
    fits, reports = read_json(fit_path / 'summary.json'), read_json(eval_path / 'summary.json')
    expected = {f"{dataset}__{variant['name']}__seed{seed}" +
                ('__independent_columns' if control != 'original' else '')
                for dataset in protocol['datasets'] for variant in protocol['variants']
                for control, seeds in protocol['controls'].items() for seed in seeds}
    assert len(fits) == len(reports) == len(expected) == 225
    assert {r['path'] for r in fits} == {r['arm'] for r in reports} == expected
    assert eval_seal['fit_arms_checked'] == 225
    manifest_path = (STUDY / protocol['data']).resolve() / 'manifest.json'
    assert sha(manifest_path) == protocol['data_manifest_sha256'] == fit_seal['data_manifest_sha256'] == eval_seal['data_manifest_sha256']
    return protocol, reports, read_json(manifest_path), fit_seal, eval_seal


def main():
    protocol, reports, manifest, fit_seal, eval_seal = require_seals()
    public, references = {}, {}
    assays = protocol['assays']
    for dataset in protocol['datasets']:
        for assay in assays:
            item = manifest['datasets'][dataset]['assays'][assay]['cohorts']['test']
            path = REPO / item['path']
            assert sha(path) == item['sha256'] == eval_seal['consumed_input_sha256'][item['path']]
            public[dataset, assay] = load(path)
        item = manifest['evaluator'][dataset]
        path = REPO / item['path']
        assert sha(path) == item['sha256'] == eval_seal['consumed_input_sha256'][item['path']]
        references[dataset] = load(path)

    real_reference = references['scmtr_seq_s1']
    assert len(set(real_reference['canonical_ids'])) == 595
    joined, columns = [], []
    for assay in assays:
        source = public['scmtr_seq_s1', assay]
        order = real_reference[f'paired_indices_{assay}']
        np.testing.assert_array_equal(np.sort(order), np.arange(595))
        joined.append(source['raw_counts'][order])
        columns.extend(source['feature_ids'])
    np.testing.assert_array_equal(np.column_stack(joined), real_reference['raw_counts'])
    np.testing.assert_array_equal(columns, real_reference['column_ids'])
    np.testing.assert_array_equal(real_reference['column_assays'], np.repeat(assays, 300))

    n, repetitions = 595, protocol['evaluation']['paired_bootstrap']['replicates']
    assert repetitions == 200 and protocol['evaluation']['paired_bootstrap']['seed'] == 3000
    rows = np.random.default_rng(3000).integers(0, n, size=(repetitions, n))
    weights = np.array([np.bincount(row, minlength=n) for row in rows], float)
    rate_cache, covariance_cache, literal_cache = {}, {}, {}
    counts = dict(arms=0, failed_arms=0, within_assay_views=0, real_pair_level_views=0,
                  unscorable_scale_views=0, state_correlation_matrices=0,
                  literal_bootstrap_model_and_zero_checks=0, bootstrap_error_identities=0,
                  bootstrap_summary_checks=0, nonzero_prediction_views=0)
    aggregate = dict(moment_pairs=0, confirmed_pairs=0, moment_all_assay=0, confirmed_all_assay=0)
    for report in reports:
        name, dataset = report['arm'], report['dataset']
        fit = read_json(STUDY / 'runs_v1' / name / 'result.json')
        assert report == read_json(STUDY / 'evaluation_v1' / name / 'result.json')
        assert report['fit_result_sha256'] == sha(STUDY / 'runs_v1' / name / 'result.json')
        assert report['causal_recovery_claim'] is False
        assert report['variant'] == fit['variant'] and report['source_seed'] == fit['source_seed'] and report['control'] == fit['control']
        counts['arms'] += 1
        if fit['status'] == 'failed':
            assert report['evaluation_status'] == 'fit_failed'
            counts['failed_arms'] += 1
            continue
        assert report['evaluation_status'] == 'complete'
        model = load(STUDY / 'runs_v1' / name / 'arrays.npz')
        measured = load(STUDY / 'evaluation_v1' / name / 'arrays.npz')
        assert report['arrays_sha256'] == sha(STUDY / 'evaluation_v1' / name / 'arrays.npz')
        for level in ['moment', 'confirmed']:
            for group in ['pair', 'all_assay']:
                value = len(fit[f'{level}_{group}_matches'])
                assert report[f'{level}_{group}_match_count'] == value
                aggregate[f'{level}_' + ('pairs' if group == 'pair' else 'all_assay')] += value
        width = fit['variant']['width']
        views = [('full_panel', width), ('original30_anchor', 30)]
        rates, rate_keys = {}, {}
        metadata = {a['assay']: a for a in fit['assays']}
        for assay in assays:
            source = public[dataset, assay]
            parameters = fit['exposure_parameters'][assay]
            key = (dataset, assay, json.dumps(parameters, sort_keys=True))
            rate_keys[assay] = key
            if key not in rate_cache:
                y = source['raw_counts'].astype(float)
                assert y.shape[1] == 300
                outside = source['library_depth'] - y.sum(1)
                exposure = np.ones(len(y)) if parameters['mode'] == 'unit' else (outside / parameters['normalization']) ** parameters['beta']
                assert np.all(exposure > 0) and np.isfinite(exposure).all()
                rate_cache[key] = y / exposure[:, None]
                covariance_cache[key] = cov(rate_cache[key])
            rates[assay] = rate_cache[key][:, :width]
            mixing = model[f'mixing_{assay}']
            prediction = mixing @ mixing.T + np.diag(model[f'noise_diagonal_{assay}'])
            if metadata[assay]['status'] == 'ok':
                close(prediction, model[f'predicted_observed_covariance_{assay}'])
            else:
                assert np.isnan(model[f'predicted_observed_covariance_{assay}']).all()
                prediction[:] = np.nan
            for view, p in views:
                record = report['within_assay'][assay]['views'][view]
                scale = model[f'scale_{assay}'][:p]
                if not np.isfinite(scale).all() or np.any(scale <= 0):
                    assert record['status'] == 'unscorable'
                    counts['unscorable_scale_views'] += 1
                    continue
                truth = covariance_cache[key][:p, :p] / np.outer(scale, scale)
                predicted = prediction[:p, :p] / np.outer(scale, scale)
                diagonal, off = np.eye(p), ~np.eye(p, dtype=bool)
                assert record['features'] == p
                assert record['status'] == ('ok' if np.isfinite(predicted).all() else 'model_abstained')
                errors(record['model_full'], predicted, truth)
                errors(record['model_offdiagonal'], predicted, truth, off)
                errors(record['diagonal_training_baseline_full'], diagonal, truth)
                errors(record['diagonal_training_baseline_offdiagonal'], diagonal, truth, off)
                close(record['standardized_mean_rmse'], np.linalg.norm((rates[assay][:, :p].mean(0) - model[f'mean_{assay}'][:p]) / scale) / np.sqrt(p))
                counts['within_assay_views'] += 1

        if dataset != 'scmtr_seq_s1':
            reference = references[dataset]
            for position, assay in enumerate(assays):
                np.testing.assert_array_equal(public[dataset, assay]['cell_ids'], reference[f'ids_{assay}'])
                score = (rates[assay] - model[f'mean_{assay}']) @ model[f'unmixing_{assay}'].T
                k = score.shape[1]
                with np.errstate(divide='ignore', invalid='ignore'):
                    matrix = (np.corrcoef(np.column_stack([score, reference[f'state_{assay}']]), rowvar=False)[:k, k:]
                              if k else np.empty((0, 2)))
                close(matrix, measured[f'state_correlations_all_{assay}'])
                state_report = report['simulation_state'][assay]
                summary_correlations(state_report['all_sources'], matrix)
                counts['state_correlation_matrices'] += 1
                for level in ['moment', 'confirmed']:
                    pair_indices = sorted({m['ia'] if m['a'] == assay else m['ib'] for m in fit[f'{level}_pair_matches'] if assay in [m['a'], m['b']]})
                    all_indices = sorted({m['indices'][position] for m in fit[f'{level}_all_assay_matches']})
                    record = state_report['levels'][level]
                    assert record['pair_matched_indices'] == pair_indices and record['all_assay_matched_indices'] == all_indices
                    for label, indices in [('pair', pair_indices), ('all_assay', all_indices)]:
                        close(matrix[indices], measured[f'state_correlations_{level}_{label}_{assay}'])
                        summary_correlations(record[f'{label}_matched'], matrix[indices])
                        counts['state_correlation_matrices'] += 1
            continue

        reference = references[dataset]
        for a, b in itertools.combinations(assays, 2):
            pair = report['real_cross_assay'][f'{a}__{b}']
            assert pair['paired_cells'] == n
            for endpoint in [a, b]:
                assert pair['endpoint_source_fits'][endpoint] == {key: metadata[endpoint].get(key) for key in ['status', 'reason', 'k']}
            pair_key = (rate_keys[a], rate_keys[b])
            left = rate_cache[rate_keys[a]][reference[f'paired_indices_{a}']]
            right = rate_cache[rate_keys[b]][reference[f'paired_indices_{b}']]
            if pair_key not in covariance_cache:
                covariance_cache[pair_key] = cov(left, right)
                literal_cache[pair_key] = {i: cov(left[rows[i]], right[rows[i]]) for i in LITERAL_RESAMPLES}
            for level in ['moment', 'confirmed']:
                record = pair['levels'][level]
                matches = [m for m in fit[f'{level}_pair_matches'] if (m['a'], m['b']) == (a, b)]
                assert record['accepted_pair_sources'] == len(matches)
                assert (record['zero_predictor_meaning'] is None) == bool(matches)
                ia, ib = [m['ia'] for m in matches], [m['ib'] for m in matches]
                signs = np.array([m['sign'] for m in matches])
                ba, bb = model[f'mixing_{a}'][:, ia], model[f'mixing_{b}'][:, ib]
                predicted_raw = (ba * signs) @ bb.T
                close(predicted_raw, model[f'predicted_cross_covariance_{level}_{a}_{b}'])
                for view, p in views:
                    metric = record['views'][view]
                    sa, sb = model[f'scale_{a}'][:p], model[f'scale_{b}'][:p]
                    if not np.isfinite(sa).all() or not np.isfinite(sb).all() or np.any(sa <= 0) or np.any(sb <= 0):
                        assert metric['status'] == 'unscorable'
                        counts['unscorable_scale_views'] += 1
                        continue
                    denominator = np.outer(sa, sb)
                    predicted = predicted_raw[:p, :p] / denominator
                    truth = covariance_cache[pair_key][:p, :p] / denominator
                    errors(metric['model'], predicted, truth)
                    errors(metric['zero'], np.zeros_like(truth), truth)
                    model_error = np.linalg.norm(predicted - truth) / p
                    zero_error = np.linalg.norm(truth) / p
                    close(metric['rmse_improvement_over_zero'], zero_error - model_error)
                    relative = metric['relative_rmse_improvement_over_zero']
                    if zero_error == 0:
                        assert relative is None
                    else:
                        close(relative, (zero_error - model_error) / zero_error)
                    prefix = f'bootstrap_{level}_{view}_{a}_{b}_'
                    zero = measured[prefix + 'zero_rmse']
                    model_error_boot = measured[prefix + 'model_rmse']
                    delta = measured[prefix + 'rmse_improvement_over_zero']
                    assert zero.shape == model_error_boot.shape == delta.shape == (repetitions,)
                    close(delta, zero - model_error_boot)
                    for i in LITERAL_RESAMPLES:
                        empirical = literal_cache[pair_key][i][:p, :p] / denominator
                        close(zero[i], np.linalg.norm(empirical) / p)
                        close(model_error_boot[i], np.linalg.norm(predicted - empirical) / p)
                        counts['literal_bootstrap_model_and_zero_checks'] += 1
                    x = (left[:, :p] - left[:, :p].mean(0)) / sa
                    y = (right[:, :p] - right[:, :p].mean(0)) / sb
                    inner = boot_inner_from_sources(x, y, ba[:p] / sa[:, None], bb[:p] / sb[:, None], signs, weights)
                    expected_squared = zero**2 + (np.sum(predicted**2) - 2 * inner) / p**2
                    close(model_error_boot**2, expected_squared)
                    if not matches:
                        np.testing.assert_array_equal(zero, model_error_boot)
                    boot = metric['paired_cell_bootstrap']
                    assert boot['replicates'] == repetitions and boot['fraction_is_not_a_p_value'] is True
                    for label, values in [('model_rmse', model_error_boot), ('zero_rmse', zero), ('rmse_improvement', delta)]:
                        close(boot[label + '_percentiles'], np.quantile(values, [.025, .5, .975]))
                    close(boot['fraction_positive_improvement'], np.mean(delta > 0))
                    counts['real_pair_level_views'] += 1
                    counts['bootstrap_error_identities'] += repetitions
                    counts['bootstrap_summary_checks'] += 1
                    counts['nonzero_prediction_views'] += int(np.any(predicted))

    receipt = {'status': 'passed', 'imports_learner_or_evaluator': False, 'new_fits': 0,
               'new_datasets': 0, 'counts': counts, 'aggregate_match_counts_across_arms': aggregate,
               'maximum_absolute_numeric_difference': MAX_DIFFERENCE,
               'literal_resample_indices_prespecified': list(LITERAL_RESAMPLES),
               'scope': ['All artifact hashes and both complete seals verified before opening data arrays',
                         'All point covariance/mean metrics, original30 anchors and state summaries replayed',
                         'Both cross-assay prediction levels reconstructed from signed matched loadings',
                         'Literal resamples 0,99,199 verified for every real pair/width/depth/scale scoring context',
                         'All 200 model-versus-zero squared-error identities checked with independent low-rank projections',
                         'All saved bootstrap percentile and positive-improvement summaries reconstructed'],
               'limits': ['The all200 error-identity check is conditional on saved zero norms; only the three prespecified literal resamples reconstruct those norms from raw counts here',
                          'A separate 48-fixture independent audit proves implementation parity of the general row-Gram formula, including nonzero predictors and large panels',
                          'Decision-gate replay is owned by the separate fit auditor; this readout replay does not refit or infer causal truth'],
               'source_sha256': {Path(__file__).name: sha(Path(__file__)), 'protocol.json': sha(STUDY / 'protocol.json')},
               'fit_provenance_sha256': sha(STUDY / 'runs_v1/provenance.json'),
               'evaluation_provenance_sha256': sha(STUDY / 'evaluation_v1/provenance.json')}
    target = Path(__file__).with_name('extension_metric_audit.json')
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def self_test():
    """Deterministic algebra fixture, not a simulated scientific dataset."""
    n = 11
    left = np.sin(np.arange(n * 6).reshape(n, 6) / 7)
    right = np.cos(np.arange(n * 8).reshape(n, 8) / 9)
    ba = np.arange(12).reshape(6, 2) / 11
    bb = np.arange(16).reshape(8, 2) / 13
    signs = np.array([1, -1])
    rows = np.array([(np.arange(n)**2 + k*np.arange(n) + 2*k) % n for k in range(6)]
                    + [np.zeros(n, dtype=int)])
    weights = np.array([np.bincount(row, minlength=n) for row in rows], float)
    expected = [np.sum(cov(left[ix], right[ix]) * ((ba * signs) @ bb.T)) for ix in rows]
    close(boot_inner_from_sources(left, right, ba, bb, signs, weights), expected)
    close(boot_inner_from_sources(left, right, ba[:, :0], bb[:, :0], signs[:0], weights), np.zeros(len(rows)))
    print(json.dumps({'self_test': 'passed', 'project_data_read': False, 'new_fits': 0,
                      'maximum_absolute_numeric_difference': MAX_DIFFERENCE}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    self_test() if args.self_test else main()
