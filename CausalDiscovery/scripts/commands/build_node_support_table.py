#!/usr/bin/env python3
"""Build node-level external-support annotations for graph plotting."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from _common import infer_repo_root, resolve_path


COORD_NODE_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
DEFAULT_ENCODE_SCREEN_URLS = [
    "https://downloads.wenglab.org/Registry-V3/GRCh38-cCREs.bed",
    "https://storage.googleapis.com/gcp.wenglab.org/Registry-V3/GRCh38-cCREs.bed",
]
ENCODE_PORTAL_BASE_URL = "https://www.encodeproject.org"
DEFAULT_ENCODE_PBMC_BIOSAMPLE_TERMS = ["peripheral blood mononuclear cell"]
JASPAR_BASE_URL = "https://jaspar.elixir.no"
UCSC_SEQUENCE_URL = "https://api.genome.ucsc.edu/getData/sequence"
EPERTURBDB_DATABASE_GLOBS = ["Database*.tsv*", "*database*.tsv*", "*.tsv"]
EPERTURBDB_ENHANCER_GENE_GLOBS = ["enhancer_gene*.bed", "*enhancer*gene*.bed"]
EPERTURBDB_TF_OVERLAP_GLOBS = ["TFoverlapEnh*.bed", "*TF*overlap*.bed", "*tf*overlap*.bed"]
NON_GENE_TOKENS = {
    "AS",
    "AND",
    "OR",
    "WELL",
    "ALSO",
    "NA",
    "N",
    "A",
    "NAN",
    "NEGATIVE",
    "CONTROL",
    "NEGATIVE_CONTROL",
    "NO",
    "RESULT",
}


def load_selected_nodes(pc_dir: Path) -> list[str]:
    summary_path = pc_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        cols = summary.get("selected_columns")
        if cols:
            return [str(c) for c in cols]
    selected_path = pc_dir / "selected_matrix.tsv"
    if selected_path.exists():
        df = pd.read_csv(selected_path, sep="\t", nrows=1)
        return [c for c in df.columns if c not in {"metacell_id", "sample_id", "row_id"}]
    raise FileNotFoundError(f"Could not infer selected nodes from {pc_dir}")


def parse_graph_node(node_name: str, region_lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    if node_name.startswith("expr__"):
        gene = node_name.split("__", 1)[1]
        # Strip expression-quantification suffixes so the gene metadata is the bare
        # symbol (e.g. expr__CD14_log1p -> CD14). Otherwise target-gene-aware support
        # matching (same_gene) fails against database gene symbols like "CD14".
        for suffix in ("_log1p", "_raw_counts", "_rawcounts", "_lognorm", "_norm"):
            if gene.endswith(suffix):
                gene = gene[: -len(suffix)]
                break
        return {
            "node_name": node_name,
            "node_type": "expression",
            "gene": gene,
            "region_name": "",
            "mark": "",
            "chrom": "",
            "start": "",
            "end": "",
            "interval_source": "",
        }

    if "__" in node_name:
        region_name, mark = node_name.split("__", 1)
    else:
        region_name, mark = node_name, ""

    coord_match = COORD_NODE_RE.match(region_name)
    if coord_match:
        return {
            "node_name": node_name,
            "node_type": "region",
            "gene": "",
            "region_name": region_name,
            "mark": mark,
            "chrom": coord_match.group("chrom"),
            "start": int(coord_match.group("start")),
            "end": int(coord_match.group("end")),
            "interval_source": "node_name",
        }

    region_row = region_lookup.get(region_name)
    if region_row is None:
        return {
            "node_name": node_name,
            "node_type": "region",
            "gene": "",
            "region_name": region_name,
            "mark": mark,
            "chrom": "",
            "start": "",
            "end": "",
            "interval_source": "",
        }

    return {
        "node_name": node_name,
        "node_type": "region",
        "gene": str(region_row.get("gene", "")),
        "region_name": region_name,
        "mark": mark,
        "chrom": str(region_row.get("chrom", "")),
        "start": int(region_row["start"]),
        "end": int(region_row["end"]),
        "interval_source": "locus_config",
    }


def normalize_external_table(path: Path, source_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"chrom", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_name} TSV missing required columns {sorted(missing)}: {path}")

    normalized = df.copy()
    normalized["source"] = source_name
    normalized["chrom"] = normalized["chrom"].astype(str)
    normalized["start"] = normalized["start"].astype(int)
    normalized["end"] = normalized["end"].astype(int)
    for col in ["gene", "support_label", "external_id", "biosample", "notes"]:
        if col not in normalized.columns:
            normalized[col] = ""
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized


def normalize_external_records(records: list[dict[str, object]], source_name: str) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["chrom", "start", "end", "source", "gene", "support_label", "external_id", "biosample", "notes"])
    normalized = pd.DataFrame(records)
    required = {"chrom", "start", "end"}
    missing = required.difference(normalized.columns)
    if missing:
        raise ValueError(f"{source_name} records missing required columns {sorted(missing)}")
    normalized["source"] = source_name
    normalized["chrom"] = normalized["chrom"].astype(str)
    normalized["start"] = normalized["start"].astype(int)
    normalized["end"] = normalized["end"].astype(int)
    for col in ["gene", "support_label", "external_id", "biosample", "notes"]:
        if col not in normalized.columns:
            normalized[col] = ""
        normalized[col] = normalized[col].fillna("").astype(str)
    return normalized


def normalize_chrom(value: object) -> str:
    text = clean_value(value).strip()
    if not text:
        return ""
    return text if text.startswith("chr") else f"chr{text}"


def safe_int(value: object) -> int | None:
    text = clean_value(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def split_gene_symbols(value: object) -> list[str]:
    text = clean_value(value).strip()
    if not text:
        return []
    if text.lower() in {"na", "n/a", "nan", "negative_control"}:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9.-]*", text.replace("_", " "))
    genes: list[str] = []
    for token in tokens:
        symbol = token.strip().upper()
        if not symbol or symbol in NON_GENE_TOKENS:
            continue
        if len(symbol) == 1 and not any(ch.isdigit() for ch in symbol):
            continue
        if symbol not in genes:
            genes.append(symbol)
    return genes


def first_nonblank(row: dict[str, object], columns: list[str]) -> str:
    for column in columns:
        value = clean_value(row.get(column))
        if value:
            return value
    return ""


def overlaps_query_intervals(
    chrom: str,
    start: int,
    end: int,
    queries_by_chrom: dict[str, list[tuple[int, int]]],
    min_overlap_bp: int,
) -> bool:
    return not queries_by_chrom or overlaps_any_query(chrom, start, end, queries_by_chrom, min_overlap_bp)


def dedupe_external_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    deduped: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for record in records:
        key = (
            record.get("chrom"),
            record.get("start"),
            record.get("end"),
            record.get("gene"),
            record.get("support_label"),
            record.get("external_id"),
            record.get("biosample"),
            record.get("notes"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def discover_first_file(directory: Path, patterns: list[str]) -> Path | None:
    for pattern in patterns:
        matches = sorted(path for path in directory.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def load_eperturbdb_database_records(
    path: Path,
    *,
    nodes: list[dict[str, object]],
    min_overlap_bp: int,
    max_records: int | None,
) -> pd.DataFrame:
    queries_by_chrom = interval_index(nodes)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, encoding="utf-8", encoding_errors="replace")
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        chrom = normalize_chrom(row.get("Chr"))
        start = safe_int(row.get("Start_38")) or safe_int(row.get("Start_19"))
        end = safe_int(row.get("End_38")) or safe_int(row.get("End_19"))
        if not chrom or start is None or end is None:
            continue
        if not overlaps_query_intervals(chrom, start, end, queries_by_chrom, min_overlap_bp):
            continue

        genes = split_gene_symbols(row.get("Gene (literature)")) or [""]
        enhancer_id = first_nonblank(row, ["Enhancer-id", "EnhancerDB_id", "Enhancer_atlas_id", "Fantom_id", "vista_id"])
        method = first_nonblank(row, ["method"])
        knockout = first_nonblank(row, ["Knockout"])
        effect = first_nonblank(row, ["Knockout_effect"])
        target_type = first_nonblank(row, ["Target"])
        tissue = first_nonblank(row, ["tissue type/cell type"])
        cell_line = first_nonblank(row, ["Cell_line", "cell_line"])
        paper = first_nonblank(row, ["Paper_title"])
        doi = first_nonblank(row, ["doi"])
        pmid = first_nonblank(row, ["PMID"])
        biosample = "; ".join(part for part in [tissue, cell_line] if part)
        support_label = "; ".join(part for part in [method, knockout, target_type] if part) or "ePerturbDB perturbation"
        notes = "; ".join(
            part
            for part in [
                effect,
                f"nearest_gene={first_nonblank(row, ['nearest_gene_name'])}" if first_nonblank(row, ["nearest_gene_name"]) else "",
                paper,
                doi,
                f"PMID={pmid}" if pmid else "",
            ]
            if part
        )
        for gene in genes:
            records.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "gene": gene,
                    "support_label": support_label,
                    "external_id": enhancer_id,
                    "biosample": biosample,
                    "notes": notes,
                }
            )
            if max_records is not None and len(records) >= max_records:
                return normalize_external_records(dedupe_external_records(records), "eperturbdb")
    return normalize_external_records(dedupe_external_records(records), "eperturbdb")


def load_eperturbdb_enhancer_gene_records(
    path: Path,
    *,
    nodes: list[dict[str, object]],
    min_overlap_bp: int,
    max_records: int | None,
) -> pd.DataFrame:
    queries_by_chrom = interval_index(nodes)
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False, encoding="utf-8", encoding_errors="replace")
    records: list[dict[str, object]] = []
    for row in df.to_dict(orient="records"):
        chrom = normalize_chrom(row.get("Chr"))
        start = safe_int(row.get("Start_38"))
        end = safe_int(row.get("End_38"))
        if not chrom or start is None or end is None:
            continue
        if not overlaps_query_intervals(chrom, start, end, queries_by_chrom, min_overlap_bp):
            continue
        enhancer_id = first_nonblank(row, ["Enhancer-id", "EnhancerDB_id", "Enhancer_atlas_id", "Fantom_id", "vista_id"])
        nearest_gene = first_nonblank(row, ["nearest_gene_name"])
        tissue = first_nonblank(row, ["tissue type/cell type"])
        cell_line = first_nonblank(row, ["cell_line", "Cell_line"])
        paper = first_nonblank(row, ["Paper_title"])
        doi = first_nonblank(row, ["doi"])
        pmid = first_nonblank(row, ["PMID"])
        records.append(
            {
                "chrom": chrom,
                "start": start,
                "end": end,
                "gene": "",
                "support_label": "ePerturbDB enhancer-gene annotation",
                "external_id": enhancer_id,
                "biosample": "; ".join(part for part in [tissue, cell_line] if part),
                "notes": "; ".join(
                    part
                    for part in [
                        f"nearest_gene={nearest_gene}" if nearest_gene else "",
                        first_nonblank(row, ["Knockout_effect"]),
                        paper,
                        doi,
                        f"PMID={pmid}" if pmid else "",
                    ]
                    if part
                ),
            }
        )
        if max_records is not None and len(records) >= max_records:
            break
    return normalize_external_records(dedupe_external_records(records), "eperturbdb_enhancer_gene")


def load_eperturbdb_tf_overlap_records(
    path: Path,
    *,
    nodes: list[dict[str, object]],
    min_overlap_bp: int,
    max_records: int | None,
) -> pd.DataFrame:
    queries_by_chrom = interval_index(nodes)
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            chrom = normalize_chrom(parts[0])
            start = safe_int(parts[1])
            end = safe_int(parts[2])
            if not chrom or start is None or end is None:
                continue
            if not overlaps_query_intervals(chrom, start, end, queries_by_chrom, min_overlap_bp):
                continue
            enhancer_id = parts[3]
            tf_labels = [label for label in (parts[4].split(";") if len(parts) > 4 else []) if label and label != "0"]
            records.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "gene": "",
                    "support_label": "ePerturbDB enhancer-TF overlap",
                    "external_id": enhancer_id,
                    "biosample": "ePerturbDB TF activity",
                    "notes": "TF overlaps: " + "|".join(sorted(set(tf_labels))) if tf_labels else "",
                }
            )
            if max_records is not None and len(records) >= max_records:
                break
    return normalize_external_records(dedupe_external_records(records), "eperturbdb_tf_overlap")


def load_eperturbdb_directory(
    *,
    directory: Path | None,
    database_tsv: Path | None,
    enhancer_gene_bed: Path | None,
    tf_overlap_bed: Path | None,
    nodes: list[dict[str, object]],
    min_overlap_bp: int,
    max_records: int | None,
) -> tuple[list[pd.DataFrame], list[dict[str, object]]]:
    if directory is not None:
        if database_tsv is None:
            database_tsv = discover_first_file(directory, EPERTURBDB_DATABASE_GLOBS)
        if enhancer_gene_bed is None:
            enhancer_gene_bed = discover_first_file(directory, EPERTURBDB_ENHANCER_GENE_GLOBS)
        if tf_overlap_bed is None:
            tf_overlap_bed = discover_first_file(directory, EPERTURBDB_TF_OVERLAP_GLOBS)

    source_tables: list[pd.DataFrame] = []
    source_events: list[dict[str, object]] = []
    if database_tsv is not None:
        source_tables.append(
            load_eperturbdb_database_records(
                database_tsv,
                nodes=nodes,
                min_overlap_bp=min_overlap_bp,
                max_records=max_records,
            )
        )
        source_events.append({"source": "eperturbdb", "mode": "local_database_tsv", "path": database_tsv})
    if enhancer_gene_bed is not None:
        source_tables.append(
            load_eperturbdb_enhancer_gene_records(
                enhancer_gene_bed,
                nodes=nodes,
                min_overlap_bp=min_overlap_bp,
                max_records=max_records,
            )
        )
        source_events.append({"source": "eperturbdb_enhancer_gene", "mode": "local_bed", "path": enhancer_gene_bed})
    if tf_overlap_bed is not None:
        source_tables.append(
            load_eperturbdb_tf_overlap_records(
                tf_overlap_bed,
                nodes=nodes,
                min_overlap_bp=min_overlap_bp,
                max_records=max_records,
            )
        )
        source_events.append({"source": "eperturbdb_tf_overlap", "mode": "local_bed", "path": tf_overlap_bed})
    return source_tables, source_events


def overlaps(start_a: int, end_a: int, start_b: int, end_b: int, min_overlap_bp: int) -> bool:
    return min(end_a, end_b) - max(start_a, start_b) >= min_overlap_bp


def graph_edges_path(graph_dir: Path) -> tuple[str, Path] | None:
    for method in ("pc", "fci", "dagma"):
        candidate = graph_dir / f"{method}_edges.tsv"
        if candidate.exists():
            return method, candidate
    matches = sorted(graph_dir.glob("*_edges.tsv"))
    if not matches:
        return None
    method = matches[0].stem.replace("_edges", "")
    return method, matches[0]


def build_region_lookup(locus_config_path: Path | None) -> dict[str, dict[str, object]]:
    if locus_config_path is None:
        return {}
    df = pd.read_csv(locus_config_path, sep="\t")
    required = {"region_name", "chrom", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Locus config missing required columns {sorted(missing)}: {locus_config_path}")
    return {
        str(row["region_name"]): row
        for row in df.to_dict(orient="records")
    }


def source_family(source: str) -> str:
    source = source.lower()
    if source.startswith("re2g"):
        return "re2g"
    if source.startswith("pchic"):
        return "pchic"
    if source.startswith("eqtl"):
        return "eqtl"
    if source.startswith("eperturb"):
        return "eperturbdb"
    if source.startswith("encode"):
        return "encode"
    if source.startswith("jaspar"):
        return "jaspar"
    return source


def highlight_class_for_sources(sources: set[str]) -> str:
    families = {source_family(source) for source in sources}
    if {"eperturbdb", "encode"}.issubset(families):
        return "both"
    if len(families) > 1:
        return "multi_source"
    if families == {"eperturbdb"}:
        return "eperturbdb_only"
    if families == {"encode"}:
        return "encode_only"
    if families == {"jaspar"}:
        return "jaspar_only"
    if families == {"re2g"}:
        return "re2g_only"
    if families == {"pchic"}:
        return "pchic_only"
    if families == {"eqtl"}:
        return "eqtl_only"
    return "none"


def source_name_from_url_arg(value: str) -> tuple[str, str]:
    if "=" in value:
        source, url = value.split("=", 1)
        source = source.strip().lower()
        url = url.strip()
    else:
        source = "external"
        url = value.strip()
    if not source or not url:
        raise ValueError(f"Expected SOURCE=URL or URL, got: {value}")
    return source, url


def split_csv(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    return out


def cache_path_for_url(cache_dir: Path, source: str, url: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", url).strip("_")
    if len(safe) > 150:
        safe = safe[:150]
    suffix = ".gz" if url.endswith(".gz") else ".txt"
    if safe.endswith(suffix):
        return cache_dir / f"{source}_{safe}"
    return cache_dir / f"{source}_{safe}{suffix}"


def iter_text_lines_from_url(url: str, cache_path: Path | None = None):
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        opener = gzip.open if cache_path.suffix == ".gz" else open
        with opener(cache_path, "rt", encoding="utf-8", errors="replace") as f:
            yield from f
        return

    request = Request(url, headers={"User-Agent": "Epigenomics-GEX-causal-analysis/1.0"})
    try:
        response = urlopen(request, timeout=60)
    except URLError as exc:
        raise RuntimeError(f"Could not fetch external evidence URL: {url}") from exc

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_tmp = Path(tempfile.mkstemp(prefix=cache_path.name, dir=cache_path.parent)[1])
        try:
            with response, cache_tmp.open("wb") as cache_fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    cache_fh.write(chunk)
            cache_tmp.replace(cache_path)
            opener = gzip.open if cache_path.suffix == ".gz" else open
            with opener(cache_path, "rt", encoding="utf-8", errors="replace") as f:
                yield from f
        finally:
            if cache_tmp.exists():
                cache_tmp.unlink()
        return

    with response:
        if url.endswith(".gz"):
            with gzip.GzipFile(fileobj=response) as gz:
                text = io.TextIOWrapper(gz, encoding="utf-8", errors="replace")
                yield from text
        else:
            for raw in response:
                yield raw.decode("utf-8", errors="replace")


def interval_index(nodes: list[dict[str, object]]) -> dict[str, list[tuple[int, int]]]:
    by_chrom: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for node in nodes:
        chrom = str(node.get("chrom", ""))
        if not chrom:
            continue
        try:
            by_chrom[chrom].append((int(node["start"]), int(node["end"])))
        except (TypeError, ValueError):
            continue
    for chrom in by_chrom:
        by_chrom[chrom].sort()
    return by_chrom


def overlaps_any_query(chrom: str, start: int, end: int, queries_by_chrom: dict[str, list[tuple[int, int]]], min_overlap_bp: int) -> bool:
    for query_start, query_end in queries_by_chrom.get(chrom, []):
        if query_start > end:
            break
        if overlaps(query_start, query_end, start, end, min_overlap_bp):
            return True
    return False


def fetch_encode_screen_records(
    *,
    nodes: list[dict[str, object]],
    urls: list[str],
    cache_dir: Path,
    min_overlap_bp: int,
    max_records: int | None,
) -> tuple[pd.DataFrame, str]:
    queries_by_chrom = interval_index(nodes)
    if not queries_by_chrom:
        return normalize_external_records([], "encode_screen"), ""

    last_error: Exception | None = None
    for url in urls:
        records: list[dict[str, object]] = []
        cache_path = None if max_records is not None else cache_path_for_url(cache_dir, "encode_screen", url)
        try:
            for line in iter_text_lines_from_url(url, cache_path):
                if not line or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                chrom = parts[0]
                if chrom not in queries_by_chrom:
                    continue
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                except ValueError:
                    continue
                if not overlaps_any_query(chrom, start, end, queries_by_chrom, min_overlap_bp):
                    continue
                accession = parts[3] if len(parts) > 3 else ""
                element_id = parts[4] if len(parts) > 4 else ""
                label = parts[5] if len(parts) > 5 else "cCRE"
                records.append(
                    {
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "gene": "",
                        "support_label": label,
                        "external_id": "|".join([x for x in [accession, element_id] if x]),
                        "biosample": "SCREEN Registry V3 human GRCh38",
                        "notes": f"ENCODE SCREEN cCRE; source={url}",
                    }
                )
                if max_records is not None and len(records) >= max_records:
                    break
            return normalize_external_records(records, "encode_screen"), url
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"Could not fetch ENCODE SCREEN cCREs from configured URLs: {last_error}")


def fetch_json_url(url: str) -> dict[str, object]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "Epigenomics-GEX-causal-analysis/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return {"@graph": [], "total": 0}
        raise RuntimeError(f"Could not fetch ENCODE metadata URL: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not fetch ENCODE metadata URL: {url}") from exc


def encode_portal_search_url(
    *,
    biosample_term: str,
    assay_term: str | None,
    assembly: str,
) -> str:
    params = {
        "type": "File",
        "output_type": "peaks",
        "file_format": "bed",
        "status": "released",
        "assembly": assembly,
        "biosample_ontology.term_name": biosample_term,
        "format": "json",
        "limit": "all",
    }
    if assay_term:
        params["assay_term_name"] = assay_term
    return f"{ENCODE_PORTAL_BASE_URL}/search/?{urlencode(params)}"


def fetch_encode_peak_file_metadata(
    *,
    biosample_terms: list[str],
    assay_terms: list[str],
    assembly: str,
    max_files: int,
) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    seen_accessions: set[str] = set()
    assay_filters = assay_terms or [None]
    for biosample_term in biosample_terms:
        for assay_term in assay_filters:
            url = encode_portal_search_url(
                biosample_term=biosample_term,
                assay_term=assay_term,
                assembly=assembly,
            )
            data = fetch_json_url(url)
            for row in data.get("@graph", []):
                if not isinstance(row, dict):
                    continue
                accession = str(row.get("accession", ""))
                href = str(row.get("href", ""))
                if not accession or not href or accession in seen_accessions:
                    continue
                seen_accessions.add(accession)
                metadata.append(row)
                if len(metadata) >= max_files:
                    return metadata
    return metadata


def encode_download_url(href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return f"{ENCODE_PORTAL_BASE_URL}{href}"


def target_label(file_row: dict[str, object]) -> str:
    target = file_row.get("target")
    if isinstance(target, dict):
        return str(target.get("label", ""))
    return ""


def fetch_encode_portal_peak_records(
    *,
    nodes: list[dict[str, object]],
    biosample_terms: list[str],
    assay_terms: list[str],
    assembly: str,
    cache_dir: Path,
    min_overlap_bp: int,
    max_files: int,
    max_records: int | None,
) -> tuple[pd.DataFrame, list[str]]:
    queries_by_chrom = interval_index(nodes)
    if not queries_by_chrom:
        return normalize_external_records([], "encode_pbmc"), []

    files = fetch_encode_peak_file_metadata(
        biosample_terms=biosample_terms,
        assay_terms=assay_terms,
        assembly=assembly,
        max_files=max_files,
    )
    records: list[dict[str, object]] = []
    used_files: list[str] = []
    for file_row in files:
        accession = str(file_row.get("accession", ""))
        href = str(file_row.get("href", ""))
        if not href:
            continue
        url = encode_download_url(href)
        cache_path = cache_path_for_url(cache_dir, "encode_pbmc", url)
        assay = str(file_row.get("assay_term_name", "ENCODE"))
        target = target_label(file_row)
        biosample = ""
        biosample_info = file_row.get("biosample_ontology")
        if isinstance(biosample_info, dict):
            biosample = str(biosample_info.get("term_name", ""))
        label_parts = [part for part in [assay, target, "peak"] if part]
        support_label = " ".join(label_parts)
        dataset = str(file_row.get("dataset", ""))
        file_type = str(file_row.get("file_type", "bed"))

        file_overlap_count = 0
        for line in iter_text_lines_from_url(url, cache_path):
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if chrom not in queries_by_chrom:
                continue
            try:
                start = int(parts[1])
                end = int(parts[2])
            except ValueError:
                continue
            if not overlaps_any_query(chrom, start, end, queries_by_chrom, min_overlap_bp):
                continue
            peak_id = parts[3] if len(parts) > 3 else ""
            external_id = accession if not peak_id else f"{accession}:{peak_id}"
            records.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "gene": "",
                    "support_label": support_label,
                    "external_id": external_id,
                    "biosample": f"ENCODE {biosample}".strip(),
                    "notes": (
                        f"ENCODE Portal {assembly} {file_type}; dataset={dataset}; "
                        f"file={accession}; source={url}"
                    ),
                }
            )
            file_overlap_count += 1
            if max_records is not None and len(records) >= max_records:
                break
        if file_overlap_count:
            used_files.append(accession)
        if max_records is not None and len(records) >= max_records:
            break

    return normalize_external_records(records, "encode_pbmc"), used_files


def cache_json_url(url: str, cache_path: Path) -> dict[str, object]:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_json_url(url)
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return data


def fetch_jaspar_motifs(
    *,
    matrix_ids: list[str],
    max_motifs: int,
    collection: str,
    tax_group: str,
    cache_dir: Path,
) -> list[dict[str, object]]:
    if matrix_ids:
        ids = matrix_ids[:max_motifs]
    else:
        params = {
            "collection": collection,
            "tax_group": tax_group,
            "page_size": str(max_motifs),
        }
        search_url = f"{JASPAR_BASE_URL}/api/v1/matrix/?{urlencode(params)}"
        search_cache = cache_path_for_url(cache_dir, "jaspar_search", search_url).with_suffix(".json")
        search = cache_json_url(search_url, search_cache)
        ids = [str(row.get("matrix_id", "")) for row in search.get("results", []) if isinstance(row, dict)]
        ids = [matrix_id for matrix_id in ids if matrix_id][:max_motifs]

    motifs: list[dict[str, object]] = []
    for matrix_id in ids:
        url = f"{JASPAR_BASE_URL}/api/v1/matrix/{matrix_id}/"
        detail_cache = cache_path_for_url(cache_dir, "jaspar_matrix", url).with_suffix(".json")
        detail = cache_json_url(url, detail_cache)
        pfm = detail.get("pfm")
        if not isinstance(pfm, dict):
            continue
        if not all(base in pfm for base in ["A", "C", "G", "T"]):
            continue
        motifs.append(detail)
    return motifs


def fetch_ucsc_sequence(
    *,
    genome: str,
    chrom: str,
    start: int,
    end: int,
    cache_dir: Path,
) -> str:
    params = {
        "genome": genome,
        "chrom": chrom,
        "start": str(start),
        "end": str(end),
    }
    url = f"{UCSC_SEQUENCE_URL}?{urlencode(params, safe=':')}"
    cache_path = cache_path_for_url(cache_dir, "ucsc_sequence", url).with_suffix(".json")
    data = cache_json_url(url, cache_path)
    return str(data.get("dna", "")).upper()


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1].upper()


def pwm_from_pfm(pfm: dict[str, object], pseudocount: float = 0.1) -> tuple[list[dict[str, float]], float, float]:
    bases = ["A", "C", "G", "T"]
    counts = {base: [float(v) for v in pfm[base]] for base in bases}
    width = min(len(counts[base]) for base in bases)
    pwm: list[dict[str, float]] = []
    min_score = 0.0
    max_score = 0.0
    for idx in range(width):
        column_total = sum(counts[base][idx] for base in bases) + pseudocount * len(bases)
        weights: dict[str, float] = {}
        for base in bases:
            prob = (counts[base][idx] + pseudocount) / column_total
            weights[base] = math.log2(prob / 0.25)
        pwm.append(weights)
        min_score += min(weights.values())
        max_score += max(weights.values())
    return pwm, min_score, max_score


def best_pwm_hit(seq: str, motif: dict[str, object]) -> tuple[float, int, str] | None:
    pfm = motif.get("pfm")
    if not isinstance(pfm, dict):
        return None
    pwm, min_score, max_score = pwm_from_pfm(pfm)
    width = len(pwm)
    if width == 0 or len(seq) < width or max_score <= min_score:
        return None
    best: tuple[float, int, str] | None = None
    for strand, scan_seq in [("+", seq.upper()), ("-", reverse_complement(seq))]:
        for offset in range(0, len(scan_seq) - width + 1):
            window = scan_seq[offset : offset + width]
            if any(base not in "ACGT" for base in window):
                continue
            raw_score = sum(pwm[idx][base] for idx, base in enumerate(window))
            normalized = (raw_score - min_score) / (max_score - min_score)
            if best is None or normalized > best[0]:
                genomic_offset = offset if strand == "+" else len(seq) - offset - width
                best = (normalized, genomic_offset, strand)
    return best


def fetch_jaspar_records(
    *,
    nodes: list[dict[str, object]],
    cache_dir: Path,
    genome: str,
    matrix_ids: list[str],
    max_motifs: int,
    min_score: float,
    max_peak_bp: int,
    collection: str,
    tax_group: str,
) -> tuple[pd.DataFrame, list[str]]:
    region_nodes = [
        node for node in nodes
        if str(node.get("node_type")) == "region"
        and str(node.get("chrom", ""))
        and clean_value(node.get("start"))
        and clean_value(node.get("end"))
        and int(node["end"]) - int(node["start"]) <= max_peak_bp
    ]
    if not region_nodes:
        return normalize_external_records([], "jaspar"), []

    motifs = fetch_jaspar_motifs(
        matrix_ids=matrix_ids,
        max_motifs=max_motifs,
        collection=collection,
        tax_group=tax_group,
        cache_dir=cache_dir,
    )
    records: list[dict[str, object]] = []
    used_motifs: set[str] = set()
    for node in region_nodes:
        chrom = str(node["chrom"])
        start = int(node["start"])
        end = int(node["end"])
        seq = fetch_ucsc_sequence(
            genome=genome,
            chrom=chrom,
            start=start,
            end=end,
            cache_dir=cache_dir,
        )
        if not seq:
            continue
        for motif in motifs:
            hit = best_pwm_hit(seq, motif)
            if hit is None:
                continue
            score, offset, strand = hit
            if score < min_score:
                continue
            matrix_id = str(motif.get("matrix_id", ""))
            motif_name = str(motif.get("name", matrix_id))
            used_motifs.add(matrix_id)
            records.append(
                {
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "gene": "",
                    "support_label": f"JASPAR motif {motif_name}",
                    "external_id": matrix_id,
                    "biosample": f"JASPAR {collection} {tax_group}",
                    "notes": f"JASPAR PWM max_score={score:.3f}; strand={strand}; motif_offset={offset}; genome={genome}",
                }
            )
    return normalize_external_records(records, "jaspar"), sorted(used_motifs)


def fetch_template_tsv_url(url: str, source_name: str, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_path_for_url(cache_dir, source_name, url)
    lines = list(iter_text_lines_from_url(url, cache_path))
    if not lines:
        return normalize_external_records([], source_name)
    from io import StringIO

    df = pd.read_csv(StringIO("".join(lines)), sep="\t")
    tmp = cache_path.with_suffix(cache_path.suffix + ".normalized.tsv")
    df.to_csv(tmp, sep="\t", index=False)
    return normalize_external_table(tmp, source_name)


SUMMARY_COLUMNS = [
    "node_name",
    "node_type",
    "gene",
    "region_name",
    "mark",
    "chrom",
    "start",
    "end",
    "interval_source",
    "sources",
    "source_count",
    "highlight_class",
    "matched_records",
    "support_labels",
    "external_ids",
    "biosamples",
    "notes",
]


MATCH_COLUMNS = [
    "node_name",
    "node_type",
    "match_type",
    "source",
    "support_label",
    "external_id",
    "biosample",
    "external_gene",
    "external_chrom",
    "external_start",
    "external_end",
    "notes",
]


EDGE_SUPPORT_COLUMNS = [
    "method",
    "edge_text",
    "node1",
    "endpoint1",
    "node2",
    "endpoint2",
    "validated",
    "sources",
    "source_count",
    "highlight_class",
    "supported_nodes",
    "validation_rule",
    "support_labels",
    "external_ids",
    "biosamples",
    "notes",
]


def join_list(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def source_counts(df: pd.DataFrame, column: str = "source") -> dict[str, int]:
    if df.empty or column not in df.columns:
        return {}
    counts = df[column].fillna("").astype(str).value_counts().to_dict()
    return {source: int(count) for source, count in counts.items() if source}


def write_support_report(
    *,
    report_path: Path,
    graph_dir: Path,
    args,
    summary_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    edge_support_df: pd.DataFrame,
    external_df: pd.DataFrame,
    source_events: list[dict[str, object]],
    out_tsv: Path,
    matches_tsv: Path,
    edge_support_tsv: Path,
) -> None:
    lines: list[str] = [
        "# External Support Report",
        "",
        f"- Graph directory: `{graph_dir}`",
        f"- Node support table: `{out_tsv}`",
        f"- Match table: `{matches_tsv}`",
        f"- Edge support table: `{edge_support_tsv}`",
        f"- Minimum interval overlap: `{args.min_overlap_bp}` bp",
        "",
        "## Requested Sources",
        "",
    ]
    for event in source_events:
        parts = []
        for key, value in event.items():
            if value is None or value == "" or value == []:
                continue
            parts.append(f"{key}={value}")
        lines.append(
            "- "
            + "; ".join(parts)
        )
    if not source_events:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Output Summary",
            "",
            f"- Nodes: {len(summary_df)}",
            f"- Supported nodes: {int((summary_df['source_count'] > 0).sum()) if not summary_df.empty else 0}",
            f"- Match records: {len(matches_df)}",
            f"- Edges: {len(edge_support_df)}",
            f"- Validated edges: {int(edge_support_df['validated'].astype(bool).sum()) if not edge_support_df.empty else 0}",
            "",
            "## Source Record Counts",
            "",
        ]
    )
    for source, count in sorted(source_counts(external_df).items()):
        lines.append(f"- {source}: {count} external record(s)")
    if not source_counts(external_df):
        lines.append("- none")

    lines.extend(["", "## Match Counts", ""])
    for source, count in sorted(source_counts(matches_df).items()):
        lines.append(f"- {source}: {count} node match(es)")
    if not source_counts(matches_df):
        lines.append("- none")

    lines.extend(["", "## Validated Edge Counts", ""])
    validated = edge_support_df[edge_support_df["validated"].astype(bool)] if not edge_support_df.empty else edge_support_df
    if not validated.empty:
        exploded: dict[str, int] = defaultdict(int)
        for value in validated["sources"].fillna("").astype(str):
            for source in [part for part in value.split("|") if part]:
                exploded[source] += 1
        for source, count in sorted(exploded.items()):
            lines.append(f"- {source}: {count} edge(s)")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `eperturbdb` entries come from either a template TSV/URL or the downloaded ePerturbDB Database TSV; the database parser uses hg38 enhancer coordinates and `Gene (literature)` for target-gene support.",
            "- `eperturbdb_enhancer_gene` entries come from the downloaded enhancer-gene BED and are treated as enhancer interval/nearest-gene annotation.",
            "- `eperturbdb_tf_overlap` entries come from the downloaded enhancer-TF overlap BED and are treated as enhancer interval/TF context.",
            "- `encode_pbmc` entries come from ENCODE Portal file search and downloaded BED peak files.",
            "- `encode_screen` entries come from the ENCODE SCREEN cCRE Registry BED.",
            "- `jaspar` entries come from JASPAR matrices plus UCSC genome sequence retrieval.",
            "- Peak-expression edge validation requires interval evidence on the peak endpoint plus a matching external target gene.",
            "- Peak-peak edge validation accepts interval evidence on either peak endpoint from regulatory sources such as ENCODE, JASPAR, or ePerturbDB TF overlap.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def clean_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def same_gene(left: object, right: object) -> bool:
    left_text = clean_value(left).upper()
    right_text = clean_value(right).upper()
    return bool(left_text and right_text and left_text == right_text)


def records_for_edge(
    node1: str,
    node2: str,
    node_by_name: dict[str, dict[str, object]],
    matches_by_node: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], str]:
    left = node_by_name.get(node1, {})
    right = node_by_name.get(node2, {})
    left_type = clean_value(left.get("node_type"))
    right_type = clean_value(right.get("node_type"))

    expression_node = ""
    region_nodes: list[str] = []
    if left_type == "expression":
        expression_node = node1
    elif left_type == "region":
        region_nodes.append(node1)
    if right_type == "expression":
        expression_node = node2
    elif right_type == "region":
        region_nodes.append(node2)

    if expression_node and region_nodes:
        expression_gene = clean_value(node_by_name.get(expression_node, {}).get("gene"))
        edge_records: list[dict[str, object]] = []
        for region_node in region_nodes:
            for match in matches_by_node.get(region_node, []):
                if clean_value(match.get("match_type")) != "interval":
                    continue
                external_gene = clean_value(match.get("external_gene"))
                if not external_gene or not same_gene(external_gene, expression_gene):
                    continue
                edge_records.append(match)
        return edge_records, "region_interval_with_matching_target_gene"

    if left_type == "region" and right_type == "region":
        edge_records = [
            match
            for node in [node1, node2]
            for match in matches_by_node.get(node, [])
            if clean_value(match.get("match_type")) == "interval"
        ]
        return edge_records, "region_interval_on_either_endpoint"

    if left_type == "expression" and right_type == "expression":
        edge_records = [
            match
            for node in [node1, node2]
            for match in matches_by_node.get(node, [])
            if clean_value(match.get("match_type")) == "gene"
        ]
        return edge_records, "expression_gene_match_on_either_endpoint"

    edge_records = [
        match
        for node in [node1, node2]
        for match in matches_by_node.get(node, [])
    ]
    return edge_records, "support_match_on_either_endpoint"


def build_edge_support(
    graph_dir: Path,
    node_support_df: pd.DataFrame,
    matches_df: pd.DataFrame,
) -> pd.DataFrame:
    edge_info = graph_edges_path(graph_dir)
    if edge_info is None:
        return pd.DataFrame(columns=EDGE_SUPPORT_COLUMNS)
    method, edges_path = edge_info
    edges = pd.read_csv(edges_path, sep="\t")
    if edges.empty:
        return pd.DataFrame(columns=EDGE_SUPPORT_COLUMNS)

    node_by_name = {
        str(row["node_name"]): row
        for row in node_support_df.to_dict(orient="records")
    }
    matches_by_node: dict[str, list[dict[str, object]]] = defaultdict(list)
    for match in matches_df.to_dict(orient="records"):
        matches_by_node[str(match.get("node_name", ""))].append(match)

    records: list[dict[str, object]] = []
    for row in edges.to_dict(orient="records"):
        node1 = str(row.get("node1", ""))
        node2 = str(row.get("node2", ""))
        edge_matches, validation_rule = records_for_edge(node1, node2, node_by_name, matches_by_node)
        sources: set[str] = set()
        labels: list[str] = []
        external_ids: list[str] = []
        biosamples: list[str] = []
        notes: list[str] = []
        supported_nodes: list[str] = []
        for match in edge_matches:
            supported_nodes.append(clean_value(match.get("node_name")))
            source = clean_value(match.get("source"))
            if source:
                sources.add(source)
            for bucket, key in [
                (labels, "support_label"),
                (external_ids, "external_id"),
                (biosamples, "biosample"),
                (notes, "notes"),
            ]:
                value = clean_value(match.get(key))
                if value:
                    bucket.append(value)
        records.append(
            {
                "method": method,
                "edge_text": str(row.get("edge_text", f"{node1} -- {node2}")),
                "node1": node1,
                "endpoint1": str(row.get("endpoint1", "")),
                "node2": node2,
                "endpoint2": str(row.get("endpoint2", "")),
                "validated": bool(edge_matches),
                "sources": "|".join(sorted(sources)),
                "source_count": len(sources),
                "highlight_class": highlight_class_for_sources(sources),
                "supported_nodes": "|".join(sorted(set(supported_nodes))),
                "validation_rule": validation_rule,
                "support_labels": "|".join(sorted(set(labels))),
                "external_ids": "|".join(sorted(set(external_ids))),
                "biosamples": "|".join(sorted(set(biosamples))),
                "notes": " | ".join(sorted(set(notes))),
            }
        )
    return pd.DataFrame(records, columns=EDGE_SUPPORT_COLUMNS)


def main() -> int:
    p = argparse.ArgumentParser(description="Build node-level support annotations for a saved graph directory")
    p.add_argument("--repo-root", default=str(infer_repo_root()))
    p.add_argument("--pc-dir", required=True, help="Graph output directory containing run_summary.json / selected_matrix.tsv")
    p.add_argument("--locus-config", default=None, help="Optional locus config TSV for named-region nodes")
    p.add_argument("--eperturbdb-tsv", default=None, help="TSV extract from ePerturbDB using the evaluation template columns")
    p.add_argument("--re2g-tsv", default=None, help="ENCODE-rE2G monocyte gene-link support template TSV (target-gene-aware; build with scripts/build_re2g_gene_link_support.py). Loaded as source 're2g'.")
    p.add_argument("--eperturbdb-dir", default=None, help="Directory containing downloaded ePerturbDB database/enhancer/TF files")
    p.add_argument("--eperturbdb-database-tsv", default=None, help="Downloaded ePerturbDB Database TSV/TSV.txt file")
    p.add_argument("--eperturbdb-enhancer-gene-bed", default=None, help="Downloaded ePerturbDB enhancer_gene BED file")
    p.add_argument("--eperturbdb-tf-overlap-bed", default=None, help="Downloaded ePerturbDB enhancer-TF overlap BED file")
    p.add_argument("--max-eperturbdb-records", type=int, default=None, help="Optional cap for retained local ePerturbDB overlap records per source")
    p.add_argument("--encode-screen-tsv", default=None, help="TSV extract from ENCODE SCREEN using the evaluation template columns")
    p.add_argument("--fetch-encode-screen", action="store_true", help="Fetch ENCODE SCREEN human GRCh38 cCREs online and keep overlaps with graph nodes")
    p.add_argument("--encode-screen-url", action="append", default=[], help="Override/add ENCODE SCREEN BED URL. Default: Registry V3 GRCh38 cCREs")
    p.add_argument("--fetch-encode-pbmc", action="store_true", help="Fetch ENCODE Portal GRCh38 peak files for PBMC-like biosample terms and keep overlaps with graph nodes")
    p.add_argument("--encode-biosample-term", action="append", default=[], help="ENCODE biosample term for --fetch-encode-pbmc. May be repeated or comma-separated. Default: peripheral blood mononuclear cell")
    p.add_argument("--encode-assay-term", action="append", default=[], help="Optional ENCODE assay term filter for --fetch-encode-pbmc, e.g. ChIP-seq or DNase-seq. May be repeated or comma-separated")
    p.add_argument("--encode-assembly", default="GRCh38")
    p.add_argument("--max-encode-files", type=int, default=25, help="Maximum ENCODE Portal files to download for --fetch-encode-pbmc")
    p.add_argument("--fetch-jaspar", action="store_true", help="Fetch JASPAR CORE motifs and UCSC hg38 sequences, then annotate peak nodes with motif hits")
    p.add_argument("--jaspar-matrix-id", action="append", default=[], help="JASPAR matrix ID to scan, e.g. MA0139.1. May be repeated or comma-separated. Defaults to the first --max-jaspar-motifs matching matrices")
    p.add_argument("--jaspar-collection", default="CORE")
    p.add_argument("--jaspar-tax-group", default="vertebrates")
    p.add_argument("--jaspar-genome", default="hg38")
    p.add_argument("--max-jaspar-motifs", type=int, default=25)
    p.add_argument("--jaspar-min-score", type=float, default=0.85, help="Minimum normalized PWM score in [0, 1]")
    p.add_argument("--jaspar-max-peak-bp", type=int, default=2000, help="Skip intervals longer than this for online sequence motif scans")
    p.add_argument("--eperturbdb-url", default=None, help="Online TSV URL for ePerturbDB-style evidence using the evaluation template columns")
    p.add_argument("--external-support-url", action="append", default=[], help="Additional online template TSV as SOURCE=URL")
    p.add_argument("--external-support-tsv", action="append", default=[], help="Additional LOCAL template TSV as SOURCE=PATH (e.g. pchic=Data/raw/external/pchic/pchic_mon_gene_links.tsv)")
    p.add_argument("--external-cache-dir", default="Evaluation/outputs/external/cache")
    p.add_argument("--max-online-records", type=int, default=None, help="Optional cap for online records retained after locus overlap filtering")
    p.add_argument("--min-overlap-bp", type=int, default=1)
    p.add_argument("--out-tsv", default=None, help="Default: <pc-dir>/node_support.tsv")
    p.add_argument("--matches-tsv", default=None, help="Default: <pc-dir>/node_support_matches.tsv")
    p.add_argument("--edge-support-tsv", default=None, help="Default: <pc-dir>/edge_support.tsv")
    p.add_argument("--report-md", default=None, help="Default: <pc-dir>/support_report.md")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    pc_dir = resolve_path(repo_root, args.pc_dir)
    if pc_dir is None or not pc_dir.exists():
        raise FileNotFoundError(f"Missing PC directory: {args.pc_dir}")

    locus_config_path = resolve_path(repo_root, args.locus_config)
    if locus_config_path is not None and not locus_config_path.exists():
        raise FileNotFoundError(f"Missing locus config TSV: {args.locus_config}")

    source_tables: list[pd.DataFrame] = []
    source_events: list[dict[str, object]] = []
    if args.eperturbdb_tsv:
        eperturbdb_path = resolve_path(repo_root, args.eperturbdb_tsv)
        if eperturbdb_path is None or not eperturbdb_path.exists():
            raise FileNotFoundError(f"Missing ePerturbDB TSV: {args.eperturbdb_tsv}")
        source_tables.append(normalize_external_table(eperturbdb_path, "eperturbdb"))
        source_events.append({"source": "eperturbdb", "mode": "local_tsv", "path": eperturbdb_path})
    if args.re2g_tsv:
        re2g_path = resolve_path(repo_root, args.re2g_tsv)
        if re2g_path is None or not re2g_path.exists():
            raise FileNotFoundError(f"Missing rE2G TSV: {args.re2g_tsv}")
        source_tables.append(normalize_external_table(re2g_path, "re2g"))
        source_events.append({"source": "re2g", "mode": "local_tsv", "path": re2g_path})
    if args.encode_screen_tsv:
        encode_path = resolve_path(repo_root, args.encode_screen_tsv)
        if encode_path is None or not encode_path.exists():
            raise FileNotFoundError(f"Missing ENCODE SCREEN TSV: {args.encode_screen_tsv}")
        source_tables.append(normalize_external_table(encode_path, "encode_screen"))
        source_events.append({"source": "encode_screen", "mode": "local_tsv", "path": encode_path})

    region_lookup = build_region_lookup(locus_config_path)
    nodes = load_selected_nodes(pc_dir)
    parsed_nodes = [parse_graph_node(node, region_lookup) for node in nodes]

    eperturbdb_dir = resolve_path(repo_root, args.eperturbdb_dir)
    eperturbdb_database_tsv = resolve_path(repo_root, args.eperturbdb_database_tsv)
    eperturbdb_enhancer_gene_bed = resolve_path(repo_root, args.eperturbdb_enhancer_gene_bed)
    eperturbdb_tf_overlap_bed = resolve_path(repo_root, args.eperturbdb_tf_overlap_bed)
    for label, path in [
        ("ePerturbDB directory", eperturbdb_dir),
        ("ePerturbDB database TSV", eperturbdb_database_tsv),
        ("ePerturbDB enhancer-gene BED", eperturbdb_enhancer_gene_bed),
        ("ePerturbDB TF-overlap BED", eperturbdb_tf_overlap_bed),
    ]:
        if path is not None and not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if any([eperturbdb_dir, eperturbdb_database_tsv, eperturbdb_enhancer_gene_bed, eperturbdb_tf_overlap_bed]):
        eperturb_tables, eperturb_events = load_eperturbdb_directory(
            directory=eperturbdb_dir,
            database_tsv=eperturbdb_database_tsv,
            enhancer_gene_bed=eperturbdb_enhancer_gene_bed,
            tf_overlap_bed=eperturbdb_tf_overlap_bed,
            nodes=parsed_nodes,
            min_overlap_bp=args.min_overlap_bp,
            max_records=args.max_eperturbdb_records,
        )
        source_tables.extend(eperturb_tables)
        source_events.extend(eperturb_events)

    cache_dir = resolve_path(repo_root, args.external_cache_dir)
    assert cache_dir is not None
    if args.fetch_encode_screen:
        urls = args.encode_screen_url or DEFAULT_ENCODE_SCREEN_URLS
        encode_df, encode_url = fetch_encode_screen_records(
            nodes=parsed_nodes,
            urls=urls,
            cache_dir=cache_dir,
            min_overlap_bp=args.min_overlap_bp,
            max_records=args.max_online_records,
        )
        print(f"Fetched ENCODE SCREEN overlaps from: {encode_url}")
        source_tables.append(encode_df)
        source_events.append({"source": "encode_screen", "mode": "online_bed", "url": encode_url})

    if args.fetch_encode_pbmc:
        biosample_terms = split_csv(args.encode_biosample_term) or DEFAULT_ENCODE_PBMC_BIOSAMPLE_TERMS
        assay_terms = split_csv(args.encode_assay_term)
        encode_pbmc_df, used_files = fetch_encode_portal_peak_records(
            nodes=parsed_nodes,
            biosample_terms=biosample_terms,
            assay_terms=assay_terms,
            assembly=args.encode_assembly,
            cache_dir=cache_dir,
            min_overlap_bp=args.min_overlap_bp,
            max_files=args.max_encode_files,
            max_records=args.max_online_records,
        )
        print(
            "Fetched ENCODE Portal PBMC overlaps from "
            f"{len(used_files)} file(s): {','.join(used_files) if used_files else 'none'}"
        )
        source_tables.append(encode_pbmc_df)
        source_events.append(
            {
                "source": "encode_pbmc",
                "mode": "online_encode_portal",
                "biosample_terms": join_list(biosample_terms),
                "assay_terms": join_list(assay_terms),
                "assembly": args.encode_assembly,
                "used_files": join_list(used_files),
            }
        )

    if args.fetch_jaspar:
        matrix_ids = split_csv(args.jaspar_matrix_id)
        jaspar_df, used_motifs = fetch_jaspar_records(
            nodes=parsed_nodes,
            cache_dir=cache_dir,
            genome=args.jaspar_genome,
            matrix_ids=matrix_ids,
            max_motifs=args.max_jaspar_motifs,
            min_score=args.jaspar_min_score,
            max_peak_bp=args.jaspar_max_peak_bp,
            collection=args.jaspar_collection,
            tax_group=args.jaspar_tax_group,
        )
        print(
            "Fetched JASPAR motif support from "
            f"{len(used_motifs)} motif(s): {','.join(used_motifs) if used_motifs else 'none'}"
        )
        source_tables.append(jaspar_df)
        source_events.append(
            {
                "source": "jaspar",
                "mode": "online_jaspar_ucsc",
                "matrix_ids": join_list(matrix_ids) if matrix_ids else f"first_{args.max_jaspar_motifs}",
                "used_motifs": join_list(used_motifs),
                "genome": args.jaspar_genome,
                "min_score": args.jaspar_min_score,
            }
        )

    if args.eperturbdb_url:
        source_tables.append(fetch_template_tsv_url(args.eperturbdb_url, "eperturbdb", cache_dir))
        source_events.append({"source": "eperturbdb", "mode": "online_template_tsv", "url": args.eperturbdb_url})

    for source_url in args.external_support_url:
        source_name, url = source_name_from_url_arg(source_url)
        source_tables.append(fetch_template_tsv_url(url, source_name, cache_dir))
        source_events.append({"source": source_name, "mode": "online_template_tsv", "url": url})

    for source_tsv in args.external_support_tsv:
        source_name, rel_path = source_name_from_url_arg(source_tsv)
        local_path = resolve_path(repo_root, rel_path)
        if local_path is None or not local_path.exists():
            raise FileNotFoundError(f"Missing external support TSV for source '{source_name}': {rel_path}")
        source_tables.append(normalize_external_table(local_path, source_name))
        source_events.append({"source": source_name, "mode": "local_tsv", "path": local_path})

    if not source_tables:
        raise ValueError("At least one external support TSV, URL, --fetch-encode-screen, --fetch-encode-pbmc, or --fetch-jaspar is required")

    external_df = pd.concat(source_tables, ignore_index=True)
    # Materialize the external records once; rebuilding this per node is O(nodes * records)
    # and becomes the dominant cost with large sources (e.g. the ~667k-row rE2G table).
    external_records = external_df.to_dict(orient="records")
    match_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []

    for node in parsed_nodes:
        node_sources: set[str] = set()
        node_labels: list[str] = []
        node_external_ids: list[str] = []
        node_biosamples: list[str] = []
        node_notes: list[str] = []

        for ext in external_records:
            gene_match = False
            interval_match = False

            if str(node["node_type"]) == "expression":
                if str(node["gene"]).upper() and str(ext["gene"]).upper() == str(node["gene"]).upper():
                    gene_match = True
            elif node["chrom"] != "":
                if str(ext["chrom"]) == str(node["chrom"]):
                    interval_match = overlaps(
                        int(node["start"]),
                        int(node["end"]),
                        int(ext["start"]),
                        int(ext["end"]),
                        args.min_overlap_bp,
                    )

            if not gene_match and not interval_match:
                continue

            node_sources.add(str(ext["source"]))
            if str(ext["support_label"]):
                node_labels.append(str(ext["support_label"]))
            if str(ext["external_id"]):
                node_external_ids.append(str(ext["external_id"]))
            if str(ext["biosample"]):
                node_biosamples.append(str(ext["biosample"]))
            if str(ext["notes"]):
                node_notes.append(str(ext["notes"]))

            match_type = "gene" if gene_match else "interval"
            match_records.append(
                {
                    "node_name": node["node_name"],
                    "node_type": node["node_type"],
                    "match_type": match_type,
                    "source": ext["source"],
                    "support_label": ext["support_label"],
                    "external_id": ext["external_id"],
                    "biosample": ext["biosample"],
                    "external_gene": ext["gene"],
                    "external_chrom": ext["chrom"],
                    "external_start": ext["start"],
                    "external_end": ext["end"],
                    "notes": ext["notes"],
                }
            )

        summary_records.append(
            {
                "node_name": node["node_name"],
                "node_type": node["node_type"],
                "gene": node["gene"],
                "region_name": node["region_name"],
                "mark": node["mark"],
                "chrom": node["chrom"],
                "start": node["start"],
                "end": node["end"],
                "interval_source": node["interval_source"],
                "sources": "|".join(sorted(node_sources)),
                "source_count": len(node_sources),
                "highlight_class": highlight_class_for_sources(node_sources),
                "matched_records": len(
                    [m for m in match_records if m["node_name"] == node["node_name"]]
                ),
                "support_labels": "|".join(sorted(set(node_labels))),
                "external_ids": "|".join(sorted(set(node_external_ids))),
                "biosamples": "|".join(sorted(set(node_biosamples))),
                "notes": " | ".join(sorted(set(node_notes))),
            }
        )

    out_tsv = (
        resolve_path(repo_root, args.out_tsv)
        if args.out_tsv
        else pc_dir / "node_support.tsv"
    )
    matches_tsv = (
        resolve_path(repo_root, args.matches_tsv)
        if args.matches_tsv
        else pc_dir / "node_support_matches.tsv"
    )
    edge_support_tsv = (
        resolve_path(repo_root, args.edge_support_tsv)
        if args.edge_support_tsv
        else pc_dir / "edge_support.tsv"
    )
    report_md = (
        resolve_path(repo_root, args.report_md)
        if args.report_md
        else pc_dir / "support_report.md"
    )
    assert out_tsv is not None
    assert matches_tsv is not None
    assert edge_support_tsv is not None
    assert report_md is not None
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    matches_tsv.parent.mkdir(parents=True, exist_ok=True)
    edge_support_tsv.parent.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_records, columns=SUMMARY_COLUMNS)
    matches_df = pd.DataFrame(match_records, columns=MATCH_COLUMNS)
    edge_support_df = build_edge_support(pc_dir, summary_df, matches_df)
    summary_df.to_csv(out_tsv, sep="\t", index=False)
    matches_df.to_csv(matches_tsv, sep="\t", index=False)
    edge_support_df.to_csv(edge_support_tsv, sep="\t", index=False)
    write_support_report(
        report_path=report_md,
        graph_dir=pc_dir,
        args=args,
        summary_df=summary_df,
        matches_df=matches_df,
        edge_support_df=edge_support_df,
        external_df=external_df,
        source_events=source_events,
        out_tsv=out_tsv,
        matches_tsv=matches_tsv,
        edge_support_tsv=edge_support_tsv,
    )

    print(f"Wrote: {out_tsv}")
    print(f"Wrote: {matches_tsv}")
    print(f"Wrote: {edge_support_tsv}")
    print(f"Wrote: {report_md}")
    print(f"Nodes: {len(summary_records)}")
    print(f"Supported nodes: {int((summary_df['source_count'] > 0).sum())}")
    if not edge_support_df.empty:
        print(f"Supported edges: {int(edge_support_df['validated'].astype(bool).sum())}")
    print(f"Match records: {len(match_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
