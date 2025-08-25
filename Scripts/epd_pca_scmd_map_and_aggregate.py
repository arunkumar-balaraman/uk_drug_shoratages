#!/usr/bin/env python3
"""
EPD, PCA, and SCMD mapping and aggregation pipeline.

- Reads aggregated EPD, PCA, or SCMD CSVs.
- For EPD/PCA:
  - Dynamically discovers and uses all "BNF Snomed Mapping data*.xlsx" files.
  - Backfills BNF_CODE to VMP_AMP mappings, prioritizing VMP over AMP.
  - Adds a 'VMP_AMP_TYPE' column ('VMP' or 'AMP').
- For SCMD:
  - Maps ODS codes to ICB codes using Trust lookup files.
  - Enriches with ICB/Region names from the EPD_Processed_Data file.
  - Sets VMP_AMP from VMP_Code and VMP_AMP_TYPE to 'VMP'.
- Records that cannot be mapped are saved to a separate "*_Ignored_Records.csv" file.
- For successfully mapped records, it drops original mapping keys (BNF_CODE/ODS_CODE) and ADQUSAGE.
- Re-aggregates the data by all remaining dimension columns.
- Writes final processed data and logs to the 'InterimData' directory.

Examples:
  python map_and_aggregate.py --dataset epd
  python map_and_aggregate.py --dataset pca
  python map_and_aggregate.py --dataset scmd
  python map_and_aggregate.py --dataset both
  
Author: ArunKumar - Capstone Project
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import argparse
import logging
import pandas as pd
import re
from datetime import datetime
import glob
import sys

# --- Configuration ---
SUM_COLS_USE = ["QUANTITY", "ITEMS", "TOTAL_QUANTITY", "NIC", "ACTUAL_COST"]

# --- Path Management ---
@dataclass(frozen=True)
class Paths:
    """A dataclass to hold all relevant file paths."""
    base: Path
    interim: Path
    logs: Path
    maps: Path
    epd_csv: Path
    pca_csv: Path
    scmd_csv: Path
    epd_out_csv: Path
    pca_out_csv: Path
    scmd_out_csv: Path
    epd_ignored_csv: Path
    pca_ignored_csv: Path
    scmd_ignored_csv: Path
    trust_map1: Path
    trust_map2: Path

def make_paths(base: Path) -> Paths:
    """Constructs all necessary paths based on the root directory."""
    interim = base / "InterimData"
    logs = interim / "_logs"
    maps_base = base / "SourceData"
    bnf_maps = maps_base / "BNFSnomedMapping"
    trust_maps = maps_base / "TrustMapping"

    interim.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    return Paths(
        base=base,
        interim=interim,
        logs=logs,
        maps=bnf_maps,
        epd_csv=interim / "EPD_202301_202505_agg.csv",
        pca_csv=interim / "PCA_202301_202505_agg.csv",
        scmd_csv=interim / "SCMD_202301_202505_agg.csv",
        epd_out_csv=interim / "EPD_Processed_Data.csv",
        pca_out_csv=interim / "PCA_Processed_Data.csv",
        scmd_out_csv=interim / "SCMD_Processed_Data.csv",
        epd_ignored_csv=interim / "EPD_Ignored_Records.csv",
        pca_ignored_csv=interim / "PCA_Ignored_Records.csv",
        scmd_ignored_csv=interim / "SCMD_Ignored_Records.csv",
        trust_map1=trust_maps / "etrust (Include headers).csv",
        trust_map2=trust_maps / "ect (Include headers).csv",
    )

# --- Logging Setup ---
def make_logger(logs_dir: Path, name: str) -> logging.Logger:
    """Configures a logger to output to both console and a timestamped file."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{name}_{run_id}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# --- Helper Functions ---
def norm_cols(cols: List[str]) -> List[str]:
    """Normalizes column names to a consistent UPPER_SNAKE_CASE format."""
    out = []
    for c in cols:
        c2 = re.sub(r"[^0-9A-Za-z]+", "_", str(c).strip())
        c2 = re.sub(r"_+", "_", c2).strip("_").upper()
        out.append(c2)
    return out

def pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Finds the first valid column name from a list of possible candidates."""
    cols = list(df.columns)
    for cand in candidates:
        c = re.sub(r"[^0-9A-Za-z]+", "_", str(cand).strip())
        c = re.sub(r"_+", "_", c).strip("_").upper()
        if c in cols:
            return c
    return None

def load_map_vmp_amp(xlsx_path: Path, logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Loads a BNF mapping file, extracts VMP/AMP data, and adds the type column."""
    try:
        m = pd.read_excel(xlsx_path, dtype=str)
        m.columns = norm_cols(m.columns)

        col_level = pick(m, ["VMP / VMPP/ AMP / AMPP", "VMP/VMPP/AMP/AMPP", "LEVEL", "VMP_LEVEL"])
        col_bnf = pick(m, ["BNF CODE", "BNF_CODE", "BNF"])
        col_snomed = pick(m, ["SNOMED CODE", "SNOMED_CODE", "SNOMED"])

        if not all([col_level, col_bnf, col_snomed]):
            logger.warning(f"Required columns not found in {xlsx_path.name}. Skipping file.")
            return None

        m = m.rename(columns={col_bnf: "BNF_CODE", col_snomed: "CODE"})
        m["__LEVEL__"] = m[col_level].astype(str).str.strip().str.upper()
        m = m[m["__LEVEL__"].isin(["VMP", "AMP"])].copy()

        m["__PRIORITY__"] = m["__LEVEL__"].map({"VMP": 0, "AMP": 1})
        m = m.sort_values(["BNF_CODE", "__PRIORITY__", "CODE"], na_position="last")

        g = m.drop_duplicates(subset=["BNF_CODE"], keep="first")
        g = g.rename(columns={"CODE": "VMP_AMP", "__LEVEL__": "VMP_AMP_TYPE"})
        
        return g[["BNF_CODE", "VMP_AMP", "VMP_AMP_TYPE"]].reset_index(drop=True)
    except Exception as e:
        logger.error(f"Failed to read {xlsx_path.name}: {e}")
        return None

def load_trust_mapping(file: Path, logger: logging.Logger) -> pd.DataFrame:
    """Loads a Trust CSV and standardizes it to [ODS_CODE, ICB_CODE]."""
    try:
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
        df.columns = norm_cols(df.columns)
        
        ods_col = pick(df, ["ORGANISATION_CODE", "ODSCODE", "ODS_CODE"])
        icb_col = pick(df, ["HIGH_LEVEL_HEALTH_GEOGRAPHY", "ICB_CODE"])

        if not ods_col or not icb_col:
             raise ValueError("Required ODS or ICB columns not found.")
        
        df = df.rename(columns={ods_col: "ODS_CODE", icb_col: "ICB_CODE"})
        return df[["ODS_CODE", "ICB_CODE"]].drop_duplicates("ODS_CODE")
    except Exception as e:
        logger.error(f"Failed to load trust mapping {file.name}: {e}")
        return pd.DataFrame(columns=["ODS_CODE", "ICB_CODE"])

def ensure_bnf_key(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the BNF code column is named 'BNF_CODE' and is a string."""
    if "BNFCODE" in df.columns and "BNF_CODE" not in df.columns:
        df = df.rename(columns={"BNFCODE": "BNF_CODE"})
    if "BNF_CODE" in df.columns:
        df["BNF_CODE"] = df["BNF_CODE"].astype(str).str.strip()
    return df

def discover_mapping_files(maps_dir: Path, logger: logging.Logger) -> List[Path]:
    """Finds all .xlsx mapping files and sorts them from newest to oldest."""
    files = [Path(p) for p in glob.glob(str(maps_dir / "*.xlsx"))]
    
    def keyfn(p: Path) -> str:
        m = re.search(r"(20\d{6})", p.name)
        return m.group(1) if m else "00000000"
        
    files.sort(key=keyfn, reverse=True)
    
    if not files:
        logger.error(f"No mapping .xlsx files found in: {maps_dir}")
    else:
        logger.info(f"Discovered {len(files)} mapping files (newest to oldest).")
    return files

def map_vmp_amp(df: pd.DataFrame, maps: List[Path], logger: logging.Logger) -> pd.DataFrame:
    """Iteratively fills 'VMP_AMP' and 'VMP_AMP_TYPE' for EPD/PCA."""
    work = df.copy()
    work["VMP_AMP"] = pd.NA
    work["VMP_AMP_TYPE"] = pd.NA

    for i, map_path in enumerate(maps, start=1):
        missing_mask = work["VMP_AMP"].isna()
        if not missing_mask.any():
            logger.info("All BNF codes mapped; backfill complete.")
            break

        missing_bnf = work.loc[missing_mask, ["BNF_CODE"]].drop_duplicates()
        logger.info(f"Pass {i}: attempting to map {len(missing_bnf):,} missing codes using {map_path.name}")

        map_df = load_map_vmp_amp(map_path, logger)
        if map_df is None or map_df.empty:
            continue

        fill_data = missing_bnf.merge(map_df, on="BNF_CODE", how="left").dropna(subset=["VMP_AMP"])
        
        if not fill_data.empty:
            work = work.set_index("BNF_CODE")
            fill_data = fill_data.set_index("BNF_CODE")
            work.update(fill_data, overwrite=False)
            work = work.reset_index()
            logger.info(f"Filled {len(fill_data):,} new mappings.")
            
    return work

def aggregate_data(df: pd.DataFrame, drop_cols: List[str], logger: logging.Logger) -> pd.DataFrame:
    """Drops specified columns and re-aggregates the data."""
    present_to_drop = [c for c in drop_cols if c in df.columns]
    if present_to_drop:
        logger.info(f"Dropping columns before aggregation: {present_to_drop}")
        df = df.drop(columns=present_to_drop)

    for c in SUM_COLS_USE:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    group_cols = [c for c in df.columns if c not in SUM_COLS_USE]
    logger.info(f"Grouping by {len(group_cols)} columns for final aggregation.")
    agg = df.groupby(group_cols, dropna=False, as_index=False)[SUM_COLS_USE].sum()
    logger.info(f"Aggregation complete -> {len(agg):,} final rows.")
    return agg

# --- Dataset Runners ---
def run_epd_pca(label: str, in_csv: Path, out_csv: Path, ignored_csv: Path, paths: Paths, logger: logging.Logger) -> None:
    """Runs the full pipeline for EPD or PCA."""
    if not in_csv.exists():
        logger.error(f"Input file not found, skipping: {in_csv}")
        return

    logger.info(f"--- Starting processing for {label} ---")
    df = pd.read_csv(in_csv)
    df = ensure_bnf_key(df)
    logger.info(f"Loaded {len(df):,} rows from {in_csv.name}.")

    mapping_files = discover_mapping_files(paths.maps, logger)
    if not mapping_files:
        logger.error("No mapping files found. Aborting.")
        return

    mapped_df = map_vmp_amp(df, mapping_files, logger)
    
    ignored_mask = mapped_df['VMP_AMP'].isna()
    mapped_records = mapped_df[~ignored_mask]
    ignored_records = df[ignored_mask]

    if not ignored_records.empty:
        logger.info(f"Saving {len(ignored_records):,} unmapped records to {ignored_csv.name}")
        ignored_records.to_csv(ignored_csv, index=False)

    if not mapped_records.empty:
        agg_df = aggregate_data(mapped_records, ["BNF_CODE", "ADQUSAGE"], logger)
        logger.info(f"Writing {len(agg_df):,} processed rows to {out_csv.name}")
        agg_df.to_csv(out_csv, index=False)
    else:
        logger.warning("No records were successfully mapped. No output file will be generated.")
        
    logger.info(f"--- Finished processing for {label} ---\n")

def run_scmd(paths: Paths, logger: logging.Logger) -> None:
    """Runs the full processing pipeline for SCMD."""
    logger.info("--- Starting processing for SCMD ---")
    if not paths.scmd_csv.exists():
        logger.error(f"Input file not found, skipping: {paths.scmd_csv}")
        return
    if not paths.epd_out_csv.exists():
        logger.error(f"EPD processed file for lookup not found: {paths.epd_out_csv}. Run EPD first.")
        return

    scmd = pd.read_csv(paths.scmd_csv, dtype=str, keep_default_na=False)
    scmd.columns = norm_cols(scmd.columns)
    logger.info(f"Loaded {len(scmd):,} SCMD rows.")

    # Prepare SCMD data
    if "ODSCODE" in scmd.columns and "ODS_CODE" not in scmd.columns:
        scmd = scmd.rename(columns={"ODSCODE": "ODS_CODE"})

    scmd = scmd.rename(columns={"VMP_CODE": "VMP_AMP"})
    scmd["VMP_AMP_TYPE"] = "VMP"

    # Map ODS to ICB
    trust1 = load_trust_mapping(paths.trust_map1, logger)
    trust2 = load_trust_mapping(paths.trust_map2, logger)
    
    scmd_m = scmd.merge(trust1, on="ODS_CODE", how="left")
    scmd_m = scmd_m.set_index("ODS_CODE")
    scmd_m.update(trust2.set_index("ODS_CODE"), overwrite=False)
    scmd_m = scmd_m.reset_index()

    # Separate ignored records
    ignored_mask = scmd_m['ICB_CODE'].isna()
    mapped_records = scmd_m[~ignored_mask]
    ignored_records = scmd[ignored_mask]

    if not ignored_records.empty:
        logger.info(f"Saving {len(ignored_records):,} unmapped SCMD records to {paths.scmd_ignored_csv.name}")
        ignored_records.to_csv(paths.scmd_ignored_csv, index=False)

    if not mapped_records.empty:
        # Enrich with ICB/Region names
        epd_map = pd.read_csv(paths.epd_out_csv, dtype=str, keep_default_na=False)
        icb_lookup = epd_map[["ICB_CODE", "ICB_NAME", "REGION_NAME", "REGION_CODE"]].drop_duplicates("ICB_CODE")
        scmd_enriched = mapped_records.merge(icb_lookup, on="ICB_CODE", how="left")

        agg_df = aggregate_data(scmd_enriched, ["ODS_CODE", "ADQUSAGE"], logger)

        # Reorder columns to match other datasets
        final_column_order = [
            "YEAR_MONTH", "REGION_NAME", "REGION_CODE", "ICB_NAME", "ICB_CODE",
            "VMP_AMP", "VMP_AMP_TYPE",
            "QUANTITY", "ITEMS", "TOTAL_QUANTITY", "NIC", "ACTUAL_COST"
        ]
        # Filter list to only include columns present in the dataframe
        final_column_order = [col for col in final_column_order if col in agg_df.columns]
        agg_df = agg_df[final_column_order]
        
        logger.info(f"Writing {len(agg_df):,} processed SCMD rows to {paths.scmd_out_csv.name}")
        agg_df.to_csv(paths.scmd_out_csv, index=False)
    else:
        logger.warning("No SCMD records were mapped to an ICB. No output file generated.")

    logger.info("--- Finished processing for SCMD ---\n")

# --- Command-Line Interface ---
def make_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the command line."""
    p = argparse.ArgumentParser(description="EPD, PCA, and SCMD mapping and aggregation pipeline.")
    p.add_argument("--base", type=Path, default=Path.home() / "uk_drug_shoratages", help="Base path to project directory.")
    p.add_argument("--dataset", choices=["epd", "pca", "scmd", "both"], required=True, help="Which dataset(s) to process.")
    return p

def main(argv=None):
    """Main entry point of the script."""
    args = make_parser().parse_args(argv)
    paths = make_paths(args.base)
    logger = make_logger(paths.logs, name="Data_Processing_Pipeline")

    logger.info("=== Pipeline Run Started ===")
    
    try:
        run_all = args.dataset == "both"
        if args.dataset == "epd" or run_all:
            run_epd_pca("EPD", paths.epd_csv, paths.epd_out_csv, paths.epd_ignored_csv, paths, logger)
        if args.dataset == "pca" or run_all:
            run_epd_pca("PCA", paths.pca_csv, paths.pca_out_csv, paths.pca_ignored_csv, paths, logger)
        if args.dataset == "scmd" or run_all:
            run_scmd(paths, logger)
            
    except Exception as e:
        logger.exception(f"An unhandled error occurred: {e}")
        sys.exit(1)

    logger.info("=== Pipeline Run Finished Successfully ===")

if __name__ == "__main__":
    main()
