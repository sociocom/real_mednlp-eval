"""Evaluate ADE."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi

def get_nmi_score(answer_path:Path, pred_path: Path) -> float:
    df_true = pd.read_csv(answer_path)
    validate_format(df_true)

    df_pred = pd.read_csv(pred_path)
    validate_format(df_pred)

    assert len(df_pred) == len(df_true)
    nmi_score = nmi(df_true["case"], df_pred["case"], average_method="arithmetic")

    return nmi_score

def validate_format(df:pd.DataFrame) -> None:
    HEAD_CASE_ID = 72
    TAIL_CASE_ID = 134
    assert list(df.columns) == ["id", "case"], f'File has invalid columns: {df.columns}'
    assert len(df) == (TAIL_CASE_ID - HEAD_CASE_ID + 1), f'File does not have {TAIL_CASE_ID - HEAD_CASE_ID + 1} rows: {len(df)}'
    assert list(df["id"].values) == list(np.arange(HEAD_CASE_ID, TAIL_CASE_ID + 1)), f'Invalid "id" column'
    assert np.isnan(df["case"].values).sum() == 0, f'Nan remains in "case" column'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="path to your submission CSV")
    parser.add_argument("--ref", type=Path, help="path to the reference (test) CSV")
    args = parser.parse_args()

    nmi_score = get_nmi_score(args.ref, args.csv)
    print(f"NMI score: {nmi_score}")
