"""GSAS-II gateway — table rows formatted exactly as ``DataFrame.to_csv`` wrote them (A83).

The gsasii extractors build pandas DataFrames; their report files are written
by core :mod:`powderline.reports` writers (one writer for every engine). Core's
writers leave number formatting to the caller, so this formats each cell as
pandas did (``na_rep=''``, ``float_format`` on float columns only), and the
files stay byte-identical to the ones v0.1.1 wrote.
"""

from __future__ import annotations

import pandas as pd


def pandas_csv_rows(df: pd.DataFrame, float_format: str | None = None) -> list[list[str]]:
    """``df``'s rows as ``to_csv(index=False, float_format=...)`` formats them."""
    columns = []
    for name in df.columns:
        col = df[name]
        if col.dtype.kind == "f":
            fmt = (lambda v, f=float_format: f % v) if float_format else str
            columns.append(["" if pd.isna(v) else fmt(v) for v in col])
        else:
            columns.append(["" if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v)) else str(v)
                            for v in col.astype(object)])
    return [list(row) for row in zip(*columns)] if columns else []
