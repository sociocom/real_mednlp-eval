"""Evaluate ADE."""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove annotation tags from articles")
    parser.add_argument("csv", type=Path, help="path to your submission CSV")
    parser.add_argument("--ref", type=Path, help="path to the reference (test) CSV")
    args = parser.parse_args()

    df_pred = pd.read_csv(args.csv)
    df_true = pd.read_csv(args.ref)

    assert len(df_pred) == len(df_true)

    print(classification_report(df_true["ADEval"], df_pred["ADEval"]))
