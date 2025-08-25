#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Map VMP_AMP → BNFCode (via SNOMED mappings) → BNF Presentation Details (strict),
write missing list, and drop unmapped with SCMD rationale & stats.

This version uses Arun's proven header-normalization approach for the
BNF↔SNOMED mapping files (Excel/CSV), including ".0" cleanup and
newest-first file discovery by mtime.

Author: ArunKumar - Capstone Project
"""

import logging
import sys
from pathlib import Path
import re
import csv
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG (edit paths here if needed)
# =============================================================================

# Input folders/files
INTERIM_DIR = Path("/Users/arunkumarbalaraman/uk_drug_shoratages/InterimData")
PROCESSED_DIR = Path("/Users/arunkumarbalaraman/uk_drug_shoratages/ProcessedData")
MAP_DIR     = Path("/Users/arunkumarbalaraman/uk_drug_shoratages/SourceData/BNFSnomedMapping")
DETAIL_DIR  = Path("/Users/arunkumarbalaraman/uk_drug_shoratages/SourceData/BNFCodeDetail")

EPD_FILE  = PROCESSED_DIR / "EPD_Processed_Data.csv"
PCA_FILE  = PROCESSED_DIR / "PCA_Processed_Data.csv"
SCMD_FILE = PROCESSED_DIR / "SCMD_Processed_Data.csv"

# Outputs
logs_dir = INTERIM_DIR / "_logs"
OUT_MISSING = INTERIM_DIR / "Missing_VMP_AMPtoBNFCode.csv"
# --- ADDED: Define an output file for the successfully mapped data ---
OUT_MAPPED = PROCESSED_DIR / "VMP_AMP_to_BNF_Mapped.csv"
LOG_FILE    = logs_dir / "map_vmp_to_bnf.log"

# BNF detail target columns to fetch
DETAIL_TARGET_COLS = [
    "BNF_PRESENTATION_CODE",
    "BNF_PRESENTATION",              # also copied to ProductCode
    "YEAR_MONTH",
    "BNF_CHAPTER","BNF_CHAPTER_CODE",
    "BNF_SECTION","BNF_SECTION_CODE",
    "BNF_PARAGRAPH","BNF_PARAGRAPH_CODE",
    "BNF_SUBPARAGRAPH","BNF_SUBPARAGRAPH_CODE",
    "BNF_CHEMICAL_SUBSTANCE","BNF_CHEMICAL_SUBSTANCE_CODE",
]

# =============================================================================
# LOGGING
# =============================================================================
def setup_logging() -> None:
    """Set up console and file logging."""
    # Prevent duplicate handlers if run multiple times
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception as e:
        logging.warning(f"Could not attach file logger: {e}")

# =============================================================================
# CSV / EXCEL IO (robust readers)
# =============================================================================
def safe_read_csv(path: Path, usecols=None, nrows=None) -> pd.DataFrame:
    """Robust CSV reader: try C engine, fallback to Python with sniffed delimiter."""
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig",
                           engine="c", usecols=usecols, nrows=nrows)
    except Exception:
        pass
    try:
        with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            sample = fh.read(65536)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
                sep = dialect.delimiter
            except Exception:
                sep = ","
            return pd.read_csv(fh, dtype=str, sep=sep, engine="python",
                               usecols=usecols, nrows=nrows)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return pd.DataFrame()


# =============================================================================
# SNOMED → BNF mapping helpers (Arun's working approach)
# =============================================================================
def list_mapping_files_desc(folder: Path):
    """
    Discover mapping files in MAP_DIR (non-recursive):
      - Accept .csv, .xlsx, .xls
      - Sort newest-first by modified time
    """
    exts = {".csv", ".xlsx", ".xls"}
    files = [p for p in Path(folder).glob("*") if p.suffix.lower() in exts and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def _clean_colnames(cols):
    """
    Normalize headers: lowercase, replace spaces/slashes/punct with underscores,
    collapse repeats, and strip leading/trailing underscores.
    """
    out = []
    for c in cols:
        s = str(c).strip().lower()
        s = re.sub(r"\s+|/|\\|:|\(|\)|\[|\]", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        out.append(s)
    return out

def _pick_col(cands, available):
    """Pick first matching available column from candidate patterns (exact or contains)."""
    for pat in cands:
        for col in available:
            if col == pat or pat in col:
                return col
    return None

def _to_clean_str_series(s: pd.Series):
    """
    Force to string, strip, drop trailing '.0' (Excel float artifact),
    and convert 'nan'/' ' to proper NA.
    """
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace({"nan": np.nan, "": np.nan, "None": np.nan})
    return s

def read_mapping_file(path: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns: snomed_code, bnf_code (strings).
    Uses Arun's robust normalization and column picking.
    """
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str, low_memory=False)
        else:
            df = pd.read_excel(path, dtype=str)  # first sheet
    except Exception as e:
        logging.warning(f"  -> Could not read {path.name}: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        logging.warning(f"  -> {path.name} is empty.")
        return pd.DataFrame()

    # Normalize headers
    orig_cols = list(df.columns)
    norm_cols = _clean_colnames(orig_cols)
    df.columns = norm_cols

    # Candidates based on your files ("SNOMED Code", "BNF Code", etc.)
    snomed_candidates = [
        "snomed_code", "snomed", "snomed_ct_code", "sctid",
        "dmd_product_code", "dm_d_product_code", "dm_d_product", "product_code"
    ]
    bnf_candidates = ["bnf_code", "bnfcode", "bnf"]

    snomed_key = _pick_col(snomed_candidates, df.columns)
    bnf_key    = _pick_col(bnf_candidates, df.columns)

    if snomed_key is None or bnf_key is None:
        logging.warning(f"  -> {path.name}: missing required columns; saw {list(df.columns)[:8]} ...")
        return pd.DataFrame()

    out = df[[snomed_key, bnf_key]].copy()
    out.columns = ["snomed_code", "bnf_code"]
    out["snomed_code"] = _to_clean_str_series(out["snomed_code"])
    out["bnf_code"]    = _to_clean_str_series(out["bnf_code"])

    # Drop blanks / NaNs and duplicates
    out = out.dropna(subset=["snomed_code", "bnf_code"])
    out = out[out["snomed_code"].str.len() > 0]
    out = out[out["bnf_code"].str.len() > 0]
    out = out.drop_duplicates(subset=["snomed_code"])

    return out

# =============================================================================
# BNF DETAIL header utils
# =============================================================================
def _normalize(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", name.lower())

_PRES_ALIASES = {
    "bnfpresentationcode","presentationcode","bnf_presentation_code",
    "prescode","bnfprescode","bnfcode15","bnf15"
}

def canonicalize_detail_headers(df: pd.DataFrame) -> pd.DataFrame:
    alias_map = {
        "yearmonth": "YEAR_MONTH", "yyyymm": "YEAR_MONTH", "year_month": "YEAR_MONTH",
        "period": "YEAR_MONTH", "datemm": "YEAR_MONTH", "dateyyyymm": "YEAR_MONTH",
        "bnfpresentation": "BNF_PRESENTATION", "presentation": "BNF_PRESENTATION", "presname": "BNF_PRESENTATION",
        "bnfchapter": "BNF_CHAPTER", "chaptername": "BNF_CHAPTER", "chapter": "BNF_CHAPTER",
        "bnfchaptercode": "BNF_CHAPTER_CODE", "chaptercode": "BNF_CHAPTER_CODE", "chapter_cd": "BNF_CHAPTER_CODE",
        "bnfsection": "BNF_SECTION", "sectionname": "BNF_SECTION",
        "bnfsectioncode": "BNF_SECTION_CODE", "sectioncode": "BNF_SECTION_CODE", "section_cd": "BNF_SECTION_CODE",
        "bnfparagraph": "BNF_PARAGRAPH", "paragraphname": "BNF_PARAGRAPH",
        "bnfparagraphcode": "BNF_PARAGRAPH_CODE", "paragraphcode": "BNF_PARAGRAPH_CODE", "paragraph_cd": "BNF_PARAGRAPH_CODE",
        "bnfsubparagraph": "BNF_SUBPARAGRAPH", "subparagraphname": "BNF_SUBPARAGRAPH", "subparagraph": "BNF_SUBPARAGRAPH",
        "bnfsubparagraphcode": "BNF_SUBPARAGRAPH_CODE", "subparagraphcode": "BNF_SUBPARAGRAPH_CODE", "subpara_cd": "BNF_SUBPARAGRAPH_CODE",
        "bnfchemicalsubstance": "BNF_CHEMICAL_SUBSTANCE", "chemicalsubstance": "BNF_CHEMICAL_SUBSTANCE",
        "chemical": "BNF_CHEMICAL_SUBSTANCE", "bnf_chemical_substance": "BNF_CHEMICAL_SUBSTANCE",
        "bnfchemicalsubstancecode": "BNF_CHEMICAL_SUBSTANCE_CODE", "chemicalsubstancecode": "BNF_CHEMICAL_SUBSTANCE_CODE",
        "chemicalcode": "BNF_CHEMICAL_SUBSTANCE_CODE", "chemical_cd": "BNF_CHEMICAL_SUBSTANCE_CODE",
    }
    norm_to_orig = {_normalize(c): c for c in df.columns}
    rename_map = {orig: alias_map[norm] for norm, orig in norm_to_orig.items() if norm in alias_map}
    return df.rename(columns=rename_map)

def find_presentation_code_column(df: pd.DataFrame) -> str | None:
    norm_to_orig = {_normalize(c): c for c in df.columns}
    for alias in _PRES_ALIASES:
        if alias in norm_to_orig:
            return norm_to_orig[alias]
    candidates = []
    for col in df.columns:
        s = df[col].astype(str).str.strip()
        if not s.empty:
            ratio = s.str.fullmatch(r"\d{15}").mean()
            if ratio >= 0.6:
                candidates.append((col, ratio))
    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    return None

def list_detail_files_desc(detail_dir: Path) -> list[Path]:
    return sorted(detail_dir.glob("bnf_code_current_*_version_88.csv"), reverse=True)

# =============================================================================
# PIPELINE STEPS
# =============================================================================
def load_distinct_vmp_amp() -> pd.DataFrame:
    """Load VMP_AMP/VMP_AMP_TYPE from EPD/PCA/SCMD and return de-duplicated DataFrame."""
    dfs = []
    for f in [EPD_FILE, PCA_FILE, SCMD_FILE]:
        logging.info(f"Reading {f.name}")
        df = safe_read_csv(f)
        if df.empty:
            logging.warning(f"Skipping empty or unreadable file: {f.name}")
            continue
        cols_upper = {c.upper(): c for c in df.columns}
        vmp_col = cols_upper.get("VMP_AMP")
        vtype_col = cols_upper.get("VMP_AMP_TYPE")
        if not vmp_col:
            logging.error(f"Could not find a VMP_AMP column in {f}")
            continue
        keep = [vmp_col] + ([vtype_col] if vtype_col else [])
        df_subset = df[keep].copy()
        df_subset.rename(columns={vmp_col: "VMP_AMP", vtype_col: "VMP_AMP_TYPE"}, inplace=True)
        df_subset["VMP_AMP"] = df_subset["VMP_AMP"].astype("string").str.strip()
        if "VMP_AMP_TYPE" in df_subset.columns:
            df_subset["VMP_AMP_TYPE"] = df_subset["VMP_AMP_TYPE"].astype("string")
        else:
            df_subset["VMP_AMP_TYPE"] = pd.NA
        dfs.append(df_subset[["VMP_AMP","VMP_AMP_TYPE"]])

    if not dfs:
        logging.error("No data loaded from input files. Exiting.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    distinct = combined.drop_duplicates().reset_index(drop=True)
    logging.info(f"Total distinct VMP/AMP rows: {len(distinct):,}")
    return distinct

def map_snomed_to_bnf_code(distinct_vmp_amp: pd.DataFrame) -> pd.DataFrame:
    """Map SNOMED (codes['VMP_AMP']) → BNFCode using mapping files."""
    codes = distinct_vmp_amp.copy()
    codes["VMP_AMP"] = codes["VMP_AMP"].astype("string").str.strip()
    codes["BNFCode"] = pd.NA

    all_files = list_mapping_files_desc(MAP_DIR)
    total = len(codes)
    logging.info(f"Starting SNOMED→BNFCode mapping for {total:,} rows")
    logging.info(f"Found {len(all_files)} mapping file(s) in {MAP_DIR}")

    for path in all_files:
        if codes["BNFCode"].notna().all():
            logging.info("All codes mapped; stopping early.")
            break

        logging.info(f"Processing mapping file: {path.name}")
        mapping_df = read_mapping_file(path)
        if mapping_df.empty:
            logging.info("  -> Skipping: empty or missing required columns.")
            continue

        mapping_series = mapping_df.set_index("snomed_code")["bnf_code"]
        codes["BNFCode"] = codes["BNFCode"].fillna(codes["VMP_AMP"].map(mapping_series))

        total_mapped = codes["BNFCode"].notna().sum()
        logging.info(f"  -> Progress: {total_mapped:,}/{total:,} mapped ({total_mapped/total:.1%})")

    logging.info("SNOMED→BNFCode mapping complete.")
    return codes

def map_bnf_details_strict(codes: pd.DataFrame) -> pd.DataFrame:
    """Strictly map codes.BNFCode ↔ BNF_PRESENTATION_CODE + details."""
    result_df = codes.copy()
    for col in DETAIL_TARGET_COLS + ["ProductCode"]:
        if col not in result_df.columns:
            result_df[col] = pd.NA

    files = list_detail_files_desc(DETAIL_DIR)
    total = len(result_df)
    logging.info(f"Starting BNF detail mapping for {total:,} rows (strict by BNF_PRESENTATION_CODE)")
    logging.info(f"Found {len(files)} detail file(s) in {DETAIL_DIR}")

    # Use the most recent file only for a consistent view of details
    if not files:
        logging.warning("No BNF detail files found. Skipping detail mapping.")
        return result_df

    path = files[0] # Use the newest file
    logging.info(f"Processing most recent detail file: {path.name}")
    try:
        df = safe_read_csv(path)
        if df.empty:
            raise ValueError("File is empty")
    except Exception as e:
        logging.error(f"  -> Read error for {path.name}: {e}; skipping detail mapping.")
        return result_df

    df = canonicalize_detail_headers(df)
    pres_col = find_presentation_code_column(df)
    if not pres_col:
        logging.warning("  -> No BNF_PRESENTATION_CODE column found; skipping detail mapping.")
        return result_df

    df.rename(columns={pres_col: "BNF_PRESENTATION_CODE"}, inplace=True)
    df["BNF_PRESENTATION_CODE"] = df["BNF_PRESENTATION_CODE"].astype("string").str.strip().str.zfill(15)

    if "YEAR_MONTH" not in df.columns:
        m = re.search(r"bnf_code_current_(\d{6})_version_88\.csv$", path.name)
        if m:
            df["YEAR_MONTH"] = m.group(1)

    # Add ProductCode from BNF_PRESENTATION if it exists
    if "BNF_PRESENTATION" in df.columns:
        df["ProductCode"] = df["BNF_PRESENTATION"]

    avail_cols = [c for c in DETAIL_TARGET_COLS + ["ProductCode"] if c in df.columns]
    detail_lookup = df[avail_cols].drop_duplicates(subset=["BNF_PRESENTATION_CODE"]).set_index("BNF_PRESENTATION_CODE")

    # Prepare our key for merging
    result_df["BNFCode_padded"] = result_df["BNFCode"].astype("string").str.zfill(15)

    # Merge
    final_df = result_df.merge(detail_lookup, left_on="BNFCode_padded", right_index=True, how="left", suffixes=("", "_new"))

    # Coalesce new columns into old ones
    for col in detail_lookup.columns:
        if f"{col}_new" in final_df.columns:
            final_df[col] = final_df[col].fillna(final_df[f"{col}_new"])
            final_df.drop(columns=[f"{col}_new"], inplace=True)
    
    final_df.drop(columns=["BNFCode_padded"], inplace=True)

    done = final_df["BNF_PRESENTATION_CODE"].notna().sum()
    logging.info(f"  -> Presentation details mapped: {done:,}/{total:,} ({done/total:.1%})")

    logging.info("BNF detail mapping complete.")
    return final_df


def write_missing_and_drop(codes: pd.DataFrame) -> pd.DataFrame:
    """Write rows with missing BNFCode to CSV, then drop them."""
    missing = codes[codes["BNFCode"].isna()].copy()
    if not missing.empty:
        OUT_MISSING.parent.mkdir(parents=True, exist_ok=True)
        missing.to_csv(OUT_MISSING, index=False)
        logging.info(f"Missing BNFCode rows written to: {OUT_MISSING} ({len(missing):,} rows)")

        # SCMD drop stats
        scmd_df = safe_read_csv(SCMD_FILE)
        if not scmd_df.empty:
            cols_upper = {c.upper(): c for c in scmd_df.columns}
            scmd_vmp_col = cols_upper.get("VMP_AMP")
            if scmd_vmp_col:
                scmd_df[scmd_vmp_col] = scmd_df[scmd_vmp_col].astype("string").str.strip()
                missing_vmp_set = set(missing["VMP_AMP"].dropna().astype("string"))
                scmd_dropped_rows = scmd_df[scmd_df[scmd_vmp_col].isin(missing_vmp_set)]
                unique_missing_vmp_in_scmd = scmd_dropped_rows[scmd_vmp_col].nunique()
                total_lines_dropped_in_scmd = len(scmd_dropped_rows)
                logging.info(
                    "Dropping unmapped rows with rationale: "
                    "Unmapped codes (e.g., bandages, OTC items) are often from SCMD and lack BNF mappings. "
                    f"This affects {unique_missing_vmp_in_scmd:,} unique VMP codes, corresponding to "
                    f"{total_lines_dropped_in_scmd:,} total line(s) in the SCMD dataset."
                )

    # Drop from codes (keep only mapped)
    before = len(codes)
    mapped_codes = codes.dropna(subset=["BNFCode"]).reset_index(drop=True)
    after = len(mapped_codes)
    logging.info(f"Dropped {before - after:,} unmapped row(s); {after:,} mapped rows remain.")
    return mapped_codes

# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main execution pipeline."""
    setup_logging()
    logging.info("=== Map VMP_AMP to BNF pipeline started ===")

    # 1) Build distinct VMP/AMP across EPD/PCA/SCMD
    distinct_vmp_amp = load_distinct_vmp_amp()

    # 2) Map SNOMED → BNFCode
    codes = map_snomed_to_bnf_code(distinct_vmp_amp)

    total_codes = len(codes)
    if total_codes > 0:
        mapped_so_far = codes["BNFCode"].notna().sum()
        logging.info(f"SNOMED→BNFCode summary: {mapped_so_far:,}/{total_codes:,} ({mapped_so_far/total_codes:.1%})")

    # 3) Strict BNFCode → BNF presentation details
    codes = map_bnf_details_strict(codes)

    # 4) Write missing and drop with SCMD rationale
    codes = write_missing_and_drop(codes)

    # --- ADDED: Save the final, successfully mapped data to a CSV file ---
    if not codes.empty:
        OUT_MAPPED.parent.mkdir(parents=True, exist_ok=True)
        codes.to_csv(OUT_MAPPED, index=False)
        logging.info(f"Successfully mapped data written to: {OUT_MAPPED} ({len(codes):,} rows)")
    else:
        logging.warning("Final mapped data is empty. No output file was written.")


    logging.info("=== Map VMP_AMP to BNF pipeline complete ===")

if __name__ == "__main__":
    main()
