from typing import Optional

import pandas as pd


def print_df(
    title: str,
    df: pd.DataFrame,
    max_rows: Optional[int] = None,
    max_cols: Optional[int] = None,
    digits: int = 4,
) -> None:
    print(f"\n=== {title} ===")
    float_fmt = lambda x: f"{x:.{digits}f}"
    with pd.option_context(
        "display.max_rows",
        max_rows,
        "display.max_columns",
        max_cols,
        "display.width",
        None,
    ):
        print(df)
