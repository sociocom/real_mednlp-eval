"""Evaluate ADE in a batch."""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove annotation tags from articles")
    parser.add_argument(
        "dir", type=Path, help="path to the folder contains submitted CSVs"
    )
    parser.add_argument(
        "--ref_ja", type=Path, help="path to the reference (test) CSV (Japanese)"
    )
    parser.add_argument(
        "--ref_en", type=Path, help="path to the reference (test) CSV (English)"
    )
    args = parser.parse_args()

    ref_dfs = {"ja": pd.read_csv(args.ref_ja), "en": pd.read_csv(args.ref_en)}
    ref_srs = {
        lang: df.groupby("articleID").apply(lambda g: g["ADEval"].sum() > 0).astype(int)
        for lang, df in ref_dfs.items()
    }
    result = []
    for p in args.dir.iterdir():
        if p.suffix == ".csv":
            sigs = p.stem.split("-")
            if len(sigs) == 7:
                _, _, lang, _, _, system, num = sigs
            else:
                raise ValueError(
                    f"Filename {p.name} does not match the rule: MedTxt-CR-[lang]-ADE-test-[system]-[num].csv"
                )
            df_pred = pd.read_csv(p)
            if len(df_pred) != len(ref_dfs[lang.lower()]):
                print(f"[ERROR] {p.name} has error in the format", file=sys.stderr)
                continue
            ps, rs, fs, _ = precision_recall_fscore_support(
                ref_dfs[lang.lower()]["ADEval"], df_pred["ADEval"], labels=[0, 1, 2, 3]
            )
            p_rep, r_rep, f_rep, _ = precision_recall_fscore_support(
                ref_srs[lang.lower()],
                df_pred.groupby("docID")
                .apply(lambda g: g["ADEval"].sum() > 0)
                .astype(int),
                average="binary",
            )
            result.append(
                {
                    "lang": lang.lower(),
                    "system": system,
                    "num": num,
                    "0-P": ps[0],
                    "1-P": ps[1],
                    "2-P": ps[2],
                    "3-P": ps[3],
                    "0-R": rs[0],
                    "1-R": rs[1],
                    "2-R": rs[2],
                    "3-R": rs[3],
                    "0-F": fs[0],
                    "1-F": fs[1],
                    "2-F": fs[2],
                    "3-F": fs[3],
                    "Report-P": p_rep,
                    "Report-R": r_rep,
                    "Report-F": f_rep,
                }
            )
    df_res = pd.DataFrame.from_records(result)
    df_res.sort_values(["lang", "system", "num"], inplace=True)
    df_res.to_csv("batch_evaluation_ade.csv", index=False)

    for l, ref_df in ref_dfs.items():
        print(l)
        print(ref_df["ADEval"].value_counts().reindex([0, 1, 2, 3]))
        print(ref_srs[l].value_counts().reindex([0, 1]))
