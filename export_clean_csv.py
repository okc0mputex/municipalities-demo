import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import numpy as np


def flatten_and_clean_columns(columns: pd.Index) -> List[str]:
    def as_str(value: object) -> str:
        if value is None:
            return ""
        return str(value)

    flattened: List[str] = []
    for col in columns:
        parts: List[str]
        if isinstance(col, tuple):
            parts = [p for p in (as_str(x).strip() for x in col) if p and not p.startswith("Unnamed:")]
        else:
            s = as_str(col).strip()
            parts = [] if (not s or s.startswith("Unnamed:")) else [s]
        name = " - ".join(parts).strip()
        if not name:
            name = "column"
        flattened.append(name)

    seen: Dict[str, int] = {}
    unique: List[str] = []
    for name in flattened:
        if name not in seen:
            seen[name] = 1
            unique.append(name)
        else:
            seen[name] += 1
            unique.append(f"{name} ({seen[name]})")
    return unique


def suggest_header_row_index_1based(preview_df: pd.DataFrame) -> int:
    best_row = 1
    best_score = -1
    rows_to_check = min(len(preview_df), 30)
    for i in range(rows_to_check):
        row = preview_df.iloc[i]
        score = int(row.notna().sum())
        if score > best_score:
            best_score = score
            best_row = i + 1
    return best_row


def load_preview(path: Union[str, Path], sheet_name: Optional[str] = None, nrows: int = 30) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=nrows)
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
    return df


def load_with_headers(
    path: Union[str, Path],
    sheet_name: Optional[str],
    header_row_1based: int,
    header_depth: int,
) -> pd.DataFrame:
    start0 = max(0, header_row_1based - 1)
    header_param = list(range(start0, start0 + header_depth)) if header_depth > 1 else start0
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_param)
    if isinstance(df, dict):
        first_key = next(iter(df.keys()))
        df = df[first_key]
    return df


def clean_dataframe(df: pd.DataFrame, drop_empty: bool = True) -> pd.DataFrame:
    df = df.copy()
    df.columns = flatten_and_clean_columns(df.columns)
    if drop_empty:
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
    return df


def export_all_sheets(
    xlsx_path: Union[str, Path],
    out_dir: Union[str, Path],
    header_depth: int = 1,
    drop_empty: bool = True,
):
    xlsx_path = Path(xlsx_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(xlsx_path)
    for sheet in xls.sheet_names:
        preview = load_preview(xlsx_path, sheet_name=sheet)
        header_row = suggest_header_row_index_1based(preview)
        df = load_with_headers(xlsx_path, sheet, header_row, header_depth)
        df = clean_dataframe(df, drop_empty=drop_empty)
        safe_sheet = "_".join(str(sheet).split())
        out_file = out_dir / f"{xlsx_path.stem}__{safe_sheet}.csv"
        df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"Exported: {out_file} (rows={len(df)}, cols={len(df.columns)})")


def main():
    parser = argparse.ArgumentParser(description="Export cleaned CSVs from Excel sheets.")
    parser.add_argument("xlsx", type=str, help="Path to the .xlsx file")
    parser.add_argument("--out", type=str, default="clean_csv", help="Output directory")
    parser.add_argument("--header-depth", type=int, default=1, help="Header rows to combine (1-3)")
    parser.add_argument("--keep-empty", action="store_true", help="Keep empty rows/columns")
    args = parser.parse_args()

    export_all_sheets(
        xlsx_path=args.xlsx,
        out_dir=args.out,
        header_depth=max(1, min(3, int(args.header_depth))),
        drop_empty=not args.keep_empty,
    )


if __name__ == "__main__":
    main()


