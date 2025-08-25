#!/usr/bin/env python3
"""
NHS Open Data unified pipeline (EPD, PCA, SCMD) with dataset-specific schemas and logging.

- Chunked CSV reading (memory-safe).
- Dataset-specific dimension columns & renames:
    * EPD/PCA: YEAR_MONTH, REGION_NAME, REGION_CODE, ICB_NAME, ICB_CODE, BNF_CODE
    * SCMD   : YEAR_MONTH, ODSCode, VMP_Code
- Measures (common; missing ones are zero-filled):
    QUANTITY, ITEMS, TOTAL_QUANTITY, ADQUSAGE, NIC, ACTUAL_COST
- Dynamic output filename uses detected period range (YYYYMM).
- Logs per-file (row counts, columns added/dropped) and run-level totals to <BASE>/<OUTDIR>/_logs/.

Example:
  python nhs_pipeline.py --dataset epd pca scmd \
  --base "/path/to/uk_drug_shoratages" \
  --outdir "InterimData" --chunk-size 500000 --counts
  
Author: ArunKumar - Capstone Project
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple
import argparse
import glob
import json
import re
import sys
from datetime import datetime

import pandas as pd

# -------------------------
# Common measure columns
# -------------------------
SUM_COLS = ["QUANTITY", "ITEMS", "TOTAL_QUANTITY", "ADQUSAGE", "NIC", "ACTUAL_COST"]


# -------------------------
# Utility functions
# -------------------------
def _now_str() -> str:
    """Timestamp helper for log file names."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _extract_periods_from_names(files: Iterable[Path]) -> Tuple[str, str]:
    """
    Extract min/max YYYYMM from file names.
    Returns ("000000","000000") if none found.
    """
    periods: List[str] = []
    pat = re.compile(r"(?P<yyyy>\d{4})(?P<mm>0[1-9]|1[0-2])")
    for f in files:
        m = pat.search(f.name)
        if m:
            periods.append(m.group(0))
    if not periods:
        return "000000", "000000"
    periods.sort()
    return periods[0], periods[-1]


def _ensure_measures(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all measure columns exist (initialized to 0).
    """
    for c in SUM_COLS:
        if c not in df.columns:
            df.loc[:, c] = 0
    return df


def _coerce_types(df: pd.DataFrame, dim_cols: List[str]) -> pd.DataFrame:
    """
    Coerce measures to numeric and dimension columns to string dtype, avoiding
    FutureWarning about incompatible inplace dtype assignment.

    Strategy:
      - Measures: Use pd.to_numeric with error coercion and fillna(0).
      - Dimensions: Ensure all columns exist, then perform a single block-wise
                    cast to the 'string' dtype and fill missing values.
                    Assigning the result back with `df[cols] = ...` avoids
                    the warning triggered by `.loc` setters.
    """
    df = df.copy()

    # Measures -> numeric
    for c in SUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Dimensions -> string
    # Step 1: Ensure all dimension columns exist.
    for c in dim_cols:
        if c not in df.columns:
            df[c] = ""  # Initialize as empty strings

    # Step 2: Cast the block of dimension columns to string dtype and fill NAs.
    # This direct assignment avoids the FutureWarning.
    df[dim_cols] = df[dim_cols].astype("string").fillna("")

    return df


# -------------------------
# Dataset schema and normalizer
# -------------------------
@dataclass(frozen=True)
class DatasetSchema:
    """
    Schema for a dataset:
      - dims: columns used for GROUP BY
      - rename_map: source -> target column names
      - extra_fill: additional constant columns (optional)
    """
    dims: List[str]
    rename_map: Dict[str, str]
    extra_fill: Dict[str, str | int | float] | None = None

    def normalize(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize an input chunk to this schema:
          - rename to target names,
          - ensure measure and dimension columns exist,
          - enforce column order (dims first, then measures),
          - coerce types (dims -> string, measures -> numeric).
        """
        # Column rename to target names
        df = chunk.rename(columns=self.rename_map).copy()

        # Ensure measures exist
        df = _ensure_measures(df)

        # Ensure dimension columns exist; initialize as empty 'string' dtype
        for d in self.dims:
            if d not in df.columns:
                df.loc[:, d] = pd.Series("", index=df.index, dtype="string")

        # Attach fixed columns when requested
        if self.extra_fill:
            for k, v in self.extra_fill.items():
                if k not in df.columns:
                    df.loc[:, k] = v

        # Column order: dims + measures; force a copy to avoid view issues
        keep_cols = self.dims + SUM_COLS
        df = df.loc[:, keep_cols].copy()

        # Type coercion
        df = _coerce_types(df, dim_cols=self.dims)
        return df


# -------------------------
# Schemas (dataset-specific)
# -------------------------
# EPD: keep REGION_NAME / REGION_CODE in the output
EPD_SCHEMA = DatasetSchema(
    dims=["YEAR_MONTH", "REGION_NAME", "REGION_CODE", "ICB_NAME", "ICB_CODE", "BNF_CODE"],
    rename_map={
        # Some EPD files use REGIONAL_OFFICE_*; map those to REGION_*
        "REGIONAL_OFFICE_NAME": "REGION_NAME",
        "REGIONAL_OFFICE_CODE": "REGION_CODE",
    },
)

# PCA: keep REGION_NAME / REGION_CODE in the output
PCA_SCHEMA = DatasetSchema(
    dims=["YEAR_MONTH", "REGION_NAME", "REGION_CODE", "ICB_NAME", "ICB_CODE", "BNF_CODE"],
    rename_map={
        # REGION_* already present in PCA; mapping shown for clarity
        "REGION_NAME": "REGION_NAME",
        "REGION_CODE": "REGION_CODE",
        "BNF_PRESENTATION_CODE": "BNF_CODE",
    },
)

# SCMD: keep ODSCode and VMP_Code names in the output
SCMD_SCHEMA = DatasetSchema(
    dims=["YEAR_MONTH", "ODSCode", "VMP_Code"],
    rename_map={
        "ODS_CODE": "ODSCode",
        "VMP_SNOMED_CODE": "VMP_Code",
        "TOTAL_QUANITY_IN_VMP_UNIT": "TOTAL_QUANTITY",
        "INDICATIVE_COST": "ACTUAL_COST",
    },
)


# -------------------------
# Dataset configuration and file listing
# -------------------------
@dataclass(frozen=True)
class DatasetConfig:
    name: str
    src_dir: Path
    patterns: Sequence[str]
    schema: DatasetSchema

    def list_files(self) -> List[Path]:
        """List files matching the configured patterns, sorted by name."""
        files: List[Path] = []
        for pat in self.patterns:
            files += [Path(p) for p in glob.glob(str(self.src_dir / pat))]
        return sorted(set(files), key=lambda p: p.name)


# -------------------------
# Logging helpers
# -------------------------
def _log_paths(base_out: Path, dataset: str, period_lo: str, period_hi: str, run_id: str) -> Dict[str, Path]:
    """
    Build log file paths and ensure the log directory exists.
    """
    logs_dir = base_out / "_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    text_log = logs_dir / f"{dataset.upper()}_{period_lo}_{period_hi}_{run_id}.log"
    jsonl_log = logs_dir / f"{dataset.upper()}_{period_lo}_{period_hi}_{run_id}.jsonl"
    rollup_csv = logs_dir / "processing_log.csv"
    return {"dir": logs_dir, "text": text_log, "jsonl": jsonl_log, "rollup": rollup_csv}


def _append_text(path: Path, s: str) -> None:
    """Append a line to the text log."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(s)
        if not s.endswith("\n"):
            fh.write("\n")


def _append_jsonl(path: Path, obj: dict) -> None:
    """Append a JSON object (one per line) to the .jsonl log."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False))
        fh.write("\n")


def _append_rollup_csv(path: Path, row: Dict[str, str | int | float]) -> None:
    """
    Append a one-line rollup row to processing_log.csv, creating header when needed.
    """
    header_needed = not path.exists()
    import csv
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if header_needed:
            w.writeheader()
        w.writerow(row)


# -------------------------
# Aggregation logic
# -------------------------
def aggregate_dataset(
    cfg: DatasetConfig,
    out_dir: Path,
    chunk_size: int = 500_000,
    show_counts: bool = False,
) -> Tuple[Path, Path]:
    """
    Aggregate one dataset:
      - list files, detect period range,
      - normalize each chunk to dataset schema,
      - aggregate by dims (sum measures),
      - write CSV and Parquet,
      - log per-file and run-level info.
    """
    files = cfg.list_files()
    if not files:
        raise FileNotFoundError(f"[{cfg.name}] No files matched: {cfg.patterns}")

    period_lo, period_hi = _extract_periods_from_names(files)
    run_id = _now_str()
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{cfg.name.upper()}_{period_lo}_{period_hi}_agg"
    out_csv = out_dir / f"{stem}.csv"
    out_parquet = out_dir / f"{stem}.parquet"

    # Initialize logs
    logp = _log_paths(out_dir, cfg.name, period_lo, period_hi, run_id)
    _append_text(logp["text"], f"=== {cfg.name.upper()} run @ {run_id} ===")
    _append_text(logp["text"], f"Source dir : {cfg.src_dir}")
    _append_text(logp["text"], f"Patterns   : {cfg.patterns}")
    _append_text(logp["text"], f"Period     : {period_lo} → {period_hi}")
    _append_text(logp["text"], f"Chunk size : {chunk_size}\n")

    total_input_rows = 0
    global_agg = None  # will hold a DataFrame indexed by dimension columns

    for f in files:
        # Fast per-file row count for logging (header excluded)
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                file_rows = sum(1 for _ in fh) - 1
        except Exception:
            file_rows = -1
        total_input_rows += max(0, file_rows)

        # Header preview for added/dropped columns logging
        try:
            preview_cols = list(pd.read_csv(f, nrows=0).columns)
        except Exception:
            preview_cols = []

        target_cols = cfg.schema.dims + SUM_COLS
        cols_added = [c for c in target_cols if c not in preview_cols]
        cols_dropped = [c for c in preview_cols if c not in target_cols]

        _append_text(logp["text"], f"[FILE] {f.name}")
        _append_text(
            logp["text"],
            f"  raw_row_count   : {file_rows:,}" if file_rows >= 0 else "  raw_row_count   : (unknown)",
        )
        _append_text(logp["text"], f"  present_columns : {preview_cols}")
        _append_text(logp["text"], f"  columns_added   : {cols_added}")
        _append_text(logp["text"], f"  columns_dropped : {cols_dropped}")

        _append_jsonl(logp["jsonl"], {
            "run_id": run_id,
            "dataset": cfg.name.upper(),
            "file": f.name,
            "period_lo": period_lo,
            "period_hi": period_hi,
            "raw_row_count": file_rows,
            "present_columns": preview_cols,
            "columns_added": cols_added,
            "columns_dropped": cols_dropped,
        })

        # Chunked processing with normalization and pre-aggregation
        for chunk in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
            norm = cfg.schema.normalize(chunk)
            chunk_agg = (
                norm.groupby(cfg.schema.dims, dropna=False, as_index=False)[SUM_COLS]
                    .sum()
                    .set_index(cfg.schema.dims)
            )
            if global_agg is None:
                global_agg = chunk_agg
            else:
                global_agg = global_agg.add(chunk_agg, fill_value=0)

    if global_agg is None:
        raise RuntimeError(f"[{cfg.name}] No data read from chunks.")

    final_df = global_agg.reset_index()

    # Write outputs
    try:
        final_df.to_parquet(out_parquet, index=False)
        _append_text(logp["text"], f"Parquet written : {out_parquet}")
    except Exception as e:
        _append_text(logp["text"], f"Parquet skipped : {e}")
    final_df.to_csv(out_csv, index=False)
    _append_text(logp["text"], f"CSV written     : {out_csv}")

    # Aggregated row count for logs
    try:
        with open(out_csv, "r", encoding="utf-8", errors="ignore") as fh:
            agg_rows = sum(1 for _ in fh) - 1
    except Exception:
        agg_rows = -1

    # Run-level summary
    _append_text(logp["text"], "")
    _append_text(logp["text"], f"TOTAL input rows : {total_input_rows:,}")
    _append_text(
        logp["text"],
        f"Aggregated rows  : {agg_rows:,}" if agg_rows >= 0 else "Aggregated rows  : (unknown)",
    )
    _append_text(logp["text"], "=== END ===\n")

    _append_jsonl(logp["jsonl"], {
        "run_id": run_id,
        "dataset": cfg.name.upper(),
        "period_lo": period_lo,
        "period_hi": period_hi,
        "total_input_rows": total_input_rows,
        "aggregated_rows": agg_rows,
        "csv_output": str(out_csv),
        "parquet_output": str(out_parquet),
        "finished_at": _now_str(),
    })

    _append_rollup_csv(logp["rollup"], {
        "run_id": run_id,
        "dataset": cfg.name.upper(),
        "period_lo": period_lo,
        "period_hi": period_hi,
        "total_input_rows": total_input_rows,
        "aggregated_rows": agg_rows,
        "csv_output": str(out_csv),
        "parquet_output": str(out_parquet),
    })

    if show_counts:
        print(f"[{cfg.name}] TOTAL input rows : {total_input_rows:,}")
        print(f"[{cfg.name}] Aggregated rows  : {agg_rows:,}")
        print(f"[{cfg.name}] Logs → {logp['text'].parent}")

    return out_csv, out_parquet


# -------------------------
# CLI
# -------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NHS pipeline (EPD/PCA/SCMD) with dataset-specific schemas and logging."
    )
    p.add_argument("--base", type=Path, required=True, help="Base path to 'uk_drug_shoratages' directory.")
    p.add_argument("--outdir", type=str, default="InterimData", help="Output subdirectory (default: InterimData).")
    p.add_argument("--dataset", nargs="+", choices=["epd", "pca", "scmd"], required=True, help="Datasets to run.")
    p.add_argument("--chunk-size", type=int, default=500_000, help="CSV chunk size (default: 500000).")
    p.add_argument("--counts", action="store_true", help="Echo per-run totals to console.")
    return p


def main(argv: List[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    base: Path = args.base
    out_dir: Path = base / args.outdir

    cfgs: Dict[str, DatasetConfig] = {
        "pca": DatasetConfig(
            name="PCA",
            src_dir=base / "SourceData" / "PrescriptionCostAnalysis",
            patterns=("PCA_*.csv",),
            schema=PCA_SCHEMA,
        ),
        "epd": DatasetConfig(
            name="EPD",
            src_dir=base / "SourceData" / "EnglishPrescriptionData",
            patterns=("EPD_*.csv",),
            schema=EPD_SCHEMA,
        ),
        "scmd": DatasetConfig(
            name="SCMD",
            src_dir=base / "SourceData" / "SecondaryCareMedicines",
            patterns=("SCMD_PROVISIONAL_*.csv",),
            schema=SCMD_SCHEMA,
        ),
    }

    for key in args.dataset:
        cfg = cfgs[key]
        print(f"\n=== Running {cfg.name} ===")
        try:
            aggregate_dataset(cfg, out_dir=out_dir, chunk_size=args.chunk_size, show_counts=args.counts)
        except Exception as e:
            print(f"[{cfg.name}] ERROR: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()