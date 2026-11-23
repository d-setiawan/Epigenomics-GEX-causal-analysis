"""Frozen marginal detection bounds; never opens evaluator data."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
DATA = REPO / 'project_resources/research/count_extensions_2026_11_01/data_v1'
ASSAYS = ('RNA', 'H3K27ac', 'H3K4me3')
DATASETS = ('scmtr_seq_s1', 'linear_chain_0', 'constrained_collider_0')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts_and_ids(path, expected):
    with np.load(path, allow_pickle=False) as f:
        x, ids = f['raw_counts'], f['feature_ids']
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != len(ids) or len(ids) != 300:
            raise ValueError('invalid count or feature shape')
        if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any():
            raise ValueError('counts must be finite nonnegative integers')
        if ids.ndim != 1 or len(set(ids.tolist())) != len(ids) or ids.tolist() != expected:
            raise ValueError('feature IDs differ from manifest')
        return x[:, :30], ids[:30]


def bounds(p, q):
    if not np.isfinite([p, q]).all() or not (0 <= p <= 1 and 0 <= q <= 1):
        raise ValueError('invalid probabilities')
    lo, hi = max(0., p + q - 1), min(p, q)
    for t in (lo, hi):
        if min(t, p-t, q-t, 1-p-q+t) < -1e-12:
            raise ValueError('invalid endpoint')
    if not lo - 1e-12 <= p*q <= hi + 1e-12:
        raise ValueError('independence outside interval')
    return lo, hi


def main():
    protocol = json.loads((HERE / 'protocol.json').read_text())
    if sha(HERE / 'fit.py') != protocol['source_sha256']['fit.py'] or sha(HERE / 'evaluate.py') != protocol['source_sha256']['evaluate.py'] or sha(HERE / 'test_fit.py') != protocol['source_sha256']['test_fit.py']:
        raise ValueError('source changed after protocol seal')
    manifest_path = DATA / 'manifest.json'
    if sha(manifest_path) != protocol['manifest_sha256']:
        raise ValueError('manifest changed')
    manifest = json.loads(manifest_path.read_text())
    records, shapes = [], {}
    for dataset in DATASETS:
        loaded = {}
        for assay in ASSAYS:
            item = manifest['datasets'][dataset]['assays'][assay]
            path = DATA / 'public' / dataset / 'train' / (assay + '.npz')
            if sha(path) != item['cohorts']['train']['sha256'] or sha(path) != protocol['inputs'][str(path.relative_to(REPO))]:
                raise ValueError('input hash mismatch')
            x, ids = counts_and_ids(path, item['selected_feature_ids'])
            detection = (x > 0).mean(axis=0)
            if not np.array_equal(detection, (x[::-1] > 0).mean(axis=0)):
                raise ValueError('row reversal changed marginals')
            loaded[assay] = (detection, ids)
            shapes[dataset + '/' + assay] = list(x.shape)
        for a, b in itertools.combinations(ASSAYS, 2):
            for i, p in enumerate(loaded[a][0]):
                for j, q in enumerate(loaded[b][0]):
                    lo, hi = bounds(p, q)
                    records.append([dataset,a,b,i,j,loaded[a][1][i],loaded[b][1][j],p,q,lo,hi,p*q,lo-p*q,hi-p*q,hi-lo])
    if len(records) != 8100:
        raise ValueError('incomplete predictions')
    out = HERE / 'predictions.csv'
    with out.open('x', newline='') as f:
        w = csv.writer(f); w.writerow(['dataset','assay_a','assay_b','index_a','index_b','feature_a','feature_b','p','q','lower','upper','independent','cov_lower','cov_upper','width']); w.writerows(records)
    seal = {'protocol_sha256': sha(HERE/'protocol.json'), 'source_sha256': protocol['source_sha256'], 'inputs': protocol['inputs'], 'manifest_sha256': protocol['manifest_sha256'], 'predictions_sha256': sha(out), 'prediction_rows': len(records), 'train_shapes_first30': shapes, 'row_reversal_invariant': True, 'evaluator_read': False}
    with (HERE / 'prediction_seal.json').open('x') as f: json.dump(seal,f,indent=2,sort_keys=True)

if __name__ == '__main__': main()
