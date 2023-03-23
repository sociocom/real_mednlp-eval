"""Real-MedNLP Finer NER Evaluation.

- Partial match-aware Precision and Recall
- Weight zero/few-shot predictions (frequency in the training data)

Calculate all systems at once.
Requires IOB format (gold and pred) as input.
"""
import csv
import itertools
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm

PTN_TAG = re.compile(r"[BIO]-([3a-z\-]+)")
PTN_MOD = re.compile(r"mod=(\w+)")

TGT_TAGS = {
    "CR": ["d", "a", "timex3", "t-test", "t-key", "t-val", "m-key", "m-val"],
    "RR": ["d", "a", "timex3", "t-test"],
}


@dataclass
class Entity:
    tag: str
    mod: str
    txt: str
    start: int
    end: int = -1


def parse_label(lbl: str, tgt_tags: list[str]) -> tuple[str, str]:
    if mt := PTN_TAG.search(lbl):
        tag = mt.group(1)
        if tag not in tgt_tags:
            tag = None
    else:
        # unexpected end token is encoded like `I-/m-key`
        # multiple tag overlap is encoded like `I-m-key/B-d`
        tag = None
    if mm := PTN_MOD.search(lbl):
        mod = mm.group(1)
    else:
        mod = ""
    return tag, mod


def append_ent(ent: Entity, entities: list[Entity]) -> None:
    if len(ent.txt) != ent.end - ent.start + 1:
        ent.end = ent.start + len(ent.txt) - 1
    entities.append(ent)


def incproc_iob(
    tgt_tags: list[str],
    lineno: int,
    char: str,
    lbl: str,
    ent: Optional[Entity],
    entities: list[Entity],
) -> Optional[Entity]:
    if lbl.startswith("B"):
        # Add an Entity to the list
        if ent is not None:
            append_ent(ent, entities)

        # Init an Entity
        tag, mod = parse_label(lbl, tgt_tags)
        if tag is not None:
            ent = Entity(tag, mod, txt=char, start=lineno, end=lineno)
        else:
            ent = None
    elif lbl.startswith("I"):
        tag, mod = parse_label(lbl, tgt_tags)
        if ent is not None and ent.tag == tag:
            ent.txt += char
            ent.end = lineno
        else:  # illegal thing is happening
            # then, discard everything so far
            ent = None
    elif lbl.startswith("O"):
        if ent is not None:
            append_ent(ent, entities)
            ent = None
    else:  # not BIO scheme; discard
        ent = None

    return ent


def extract_ents_from_iob(
    iobfp: Union[str, Path], tgt_tags: list[str]
) -> tuple[list[Entity], list[Entity]]:
    entities_g: list[Entity] = []
    entities_p: list[Entity] = []
    ent_g: Optional[Entity] = None
    ent_p: Optional[Entity] = None
    skipped_rows: int = 0
    discrepancy_chars: list[str] = []
    with open(iobfp, "r") as f:
        for lineno, line in enumerate(f, 1):
            if "\t" not in line:
                if line.startswith("%"):  # sentence boundary
                    # reset skipped rows count (incremented when char_g != char_p)
                    # because overflown text of pred are discarded in IOB
                    skipped_rows = 0
                    discrepancy_chars = []
                    if ent_p:
                        append_ent(ent_p, entities_p)
                        ent_p = None
                    continue
                else:
                    raise ValueError(f"Unexpected format at L{lineno}:\n{line}")

            cols = line.strip("\n").split("\t")
            try:
                assert len(cols) == 4  # text, gold, text, pred
                char_g, lbl_g, char_p, lbl_p = cols
                # NOTE: gold/pred looks like `I-m-key mod=other` (modality included)
            except AssertionError:
                raise ValueError(f"The number of columns is not 4\n{lineno}")

            ent_g = incproc_iob(tgt_tags, lineno, char_g, lbl_g, ent_g, entities_g)
            if char_g == char_p:
                ent_p = incproc_iob(tgt_tags, lineno, char_p, lbl_p, ent_p, entities_p)
            else:
                # illegal tag sequence may be included as char (text)
                # so, just skipping such tokens may end up matching gold some time later
                skipped_rows += 1
                discrepancy_chars.append(char_g)
                if discrepancy_chars and discrepancy_chars[0] == char_p:
                    ent_p = incproc_iob(
                        tgt_tags,
                        lineno - skipped_rows,
                        char_p,
                        lbl_p,
                        ent_p,
                        entities_p,
                    )
                    discrepancy_chars.pop(0)

    return entities_g, entities_p


def make_overlap_matrix(ent_golds, ent_preds):
    shape = len(ent_golds), len(ent_preds)
    P = np.zeros(shape)
    R = np.zeros(shape)
    T = np.zeros(shape)
    M = np.zeros(shape)
    for i, ent_gold in enumerate(ent_golds):
        for j, ent_pred in enumerate(ent_preds):
            if ent_gold.start > ent_pred.end:  # [pred]-> | (gold)
                continue
            elif ent_pred.start > ent_gold.end:  # (gold) | [pred]->
                break
            else:  # overlap or match
                T[i, j] = int(ent_gold.tag == ent_pred.tag)
                M[i, j] = int(ent_gold.mod == ent_pred.mod)
                if ent_gold.start == ent_pred.start and ent_gold.end == ent_pred.end:
                    P[i, j] = 1.0
                    R[i, j] = 1.0
                else:
                    offset_l = ent_gold.start - ent_pred.start
                    offset_r = ent_pred.end - ent_gold.end
                    gold_len = len(ent_gold.txt)
                    pred_len = len(ent_pred.txt)
                    common_len = gold_len + min(0, offset_l) + min(0, offset_r)
                    R[i, j] = common_len / gold_len
                    P[i, j] = common_len / pred_len
    return np.stack((P, R, T, M))  # (layer, n_g, n_p)


def expectate(A, n):
    return A.sum() / n


def score_arr(A, B, Tag, Mod, n, w):
    """Multiple viewpoint scoring.

    # Args
        A: Matrix of Precision or Recall
        B: The other matrix of A
    """
    A_exact_ = A * B
    A_exact = np.where(A_exact_ < 1.0, np.zeros_like(A), np.ones_like(A))
    exacts = [A_exact, A_exact * Tag, A_exact * Tag * Mod]
    partials = [A, A * Tag, A * Tag * Mod]

    def weighted(arr):
        return w * arr.sum(axis=1)

    weighted_exacts = [weighted(exact) for exact in exacts]
    weighted_partials = [weighted(partial) for partial in partials]

    return [
        expectate(arr, n)
        for arr in exacts + weighted_exacts + partials + weighted_partials
    ]


def get_train_freq(trfrfp, corpus):
    TGT_TAGS[corpus]
    train_freq = pd.read_csv(trfrfp).query("tag in @tgt_tags")
    train_freq.text = train_freq.text.str.replace(" ", "")
    return train_freq.groupby("text").sum().freq


def main(dirpath: str, resume: bool = False):
    train_freqs = {  # Header = tag: str, mod: str, text: str, freq: int
        "Subtask1": {
            "CR-JA": get_train_freq(
                "weights/MedTxt-CR-JA-training-entity_stat.csv", "CR"
            ),
            "CR-EN": get_train_freq(
                "weights/MedTxt-CR-EN-training-entity_stat.csv", "CR"
            ),
            "RR-JA": get_train_freq(
                "weights/MedTxt-RR-JA-training-entity_stat.csv", "RR"
            ),
            "RR-EN": get_train_freq(
                "weights/MedTxt-RR-EN-training-entity_stat.csv", "RR"
            ),
        },
        "Subtask2": {
            "CR-JA": get_train_freq("weights/guideline-JA-entity_stat.csv", "CR"),
            "CR-EN": get_train_freq("weights/guideline-EN-entity_stat.csv", "CR"),
            "RR-JA": get_train_freq("weights/guideline-JA-entity_stat.csv", "RR"),
            "RR-EN": get_train_freq("weights/guideline-EN-entity_stat.csv", "RR"),
        },
    }
    # training term frequency ordered by test term occurrence (internal use only; just for fallback)
    # gold_weights = {
    #     "CR-JA-Subtask1": pd.read_csv("MedTxt-CR-JA-Subtask1-test_weights.csv", index_col=0).squeeze(),
    #     "CR-EN-Subtask1": pd.read_csv("MedTxt-CR-EN-Subtask1-test_weights.csv", index_col=0).squeeze(),
    #     "RR-JA-Subtask1": pd.read_csv("MedTxt-RR-JA-Subtask1-test_weights.csv", index_col=0).squeeze(),
    #     "RR-EN-Subtask1": pd.read_csv("MedTxt-RR-EN-Subtask1-test_weights.csv", index_col=0).squeeze(),
    #     "CR-JA-Subtask2": pd.read_csv("MedTxt-CR-JA-Subtask2-test_weights.csv", index_col=0).squeeze(),
    #     "CR-EN-Subtask2": pd.read_csv("MedTxt-CR-EN-Subtask2-test_weights.csv", index_col=0).squeeze(),
    #     "RR-JA-Subtask2": pd.read_csv("MedTxt-RR-JA-Subtask2-test_weights.csv", index_col=0).squeeze(),
    #     "RR-EN-Subtask2": pd.read_csv("MedTxt-RR-EN-Subtask2-test_weights.csv", index_col=0).squeeze()
    # }

    if resume:
        outf = open("finer_ner_eval-results_long.csv", "a")
        df_out = pd.read_csv(outf)
        written = df_out.groupby(
            ["corpus", "lang", "subtask", "team", "system"]
        ).groups.keys()
    else:
        outf = open("finer_ner_eval-results_long.csv", "w")
        written = []
    writer = csv.DictWriter(
        outf,
        [  # CSV header
            "corpus",
            "lang",
            "subtask",
            "team",
            "system",
            "expar",
            "tagmod",
            "weighted",
            "metric",
            "value",
        ],
    )
    writer.writeheader()
    p = Path(dirpath)
    for fp in tqdm(list(p.glob("*.iob"))):
        try:
            task, corpus, lang, _, subtask, team, system = fp.stem.split("-")
            assert task == "MedTxt", f"Task {task} is unknown."
            assert corpus in ["CR", "RR"] and lang in [
                "JA",
                "EN",
            ], f"Corpus {corpus}-{lang} is unknown."
            assert subtask in ["Subtask1", "Subtask2"], f"Unknown Subtask: {subtask}"
            if (corpus, lang, subtask, team, system) in written:
                continue
        except ValueError:
            print("\nWARNING: File name parsing failed at", fp, file=sys.stderr)
            continue

        try:
            gents, pents = extract_ents_from_iob(fp, TGT_TAGS[corpus])
        except ValueError as e:
            print("\nWARNING:", fp, "was skipped.", file=sys.stderr)
            print(e, file=sys.stderr)
            continue
        except:
            print("\nUncaught Error at:", fp, file=sys.stderr)
            raise

        # if len(gold_weights[f"{corpus}-{lang}-{subtask}"]) != len(gents):
        #     print(f"\nWARNING: {fp}", file=sys.stderr)
        #     print("#gold_entities differs:", len(gents), file=sys.stderr)
        test_ents = pd.Index([gent.txt for gent in gents])
        gold_freq = (
            train_freqs[subtask][f"{corpus}-{lang}"].reindex(test_ents).fillna(0)
        )
        gold_weighted = 1.0 / (np.log(gold_freq + 1) + 1)
        # else:
        #     gold_weighted = gold_weights[f"{corpus}-{lang}-{subtask}"]

        res = make_overlap_matrix(gents, pents)
        n_gold, n_pred = res.shape[1], res.shape[2]
        P, R, Tag, Mod = res[0], res[1], res[2], res[3]
        Ps = score_arr(P, R, Tag, Mod, n_pred, gold_weighted)
        Rs = score_arr(R, P, Tag, Mod, n_gold, gold_weighted)
        Fs = [2 * p * r / (p + r) for p, r in zip(Ps, Rs)]

        # exact/partial? | tag/mod | weighted? | metric | value | (task_meta...)
        metric = ["P", "R", "F"]
        e_or_p = ["exact", "partial"]
        weighted = ["", "weighted"]
        tagmod = ["", "tag", "tagmod"]
        # system_records.extend([])
        writer.writerows(
            [
                dict(
                    corpus=corpus,
                    lang=lang,
                    subtask=subtask,
                    team=team,
                    system=system,
                    expar=meta[1],
                    tagmod=meta[3],
                    weighted=meta[2],
                    metric=meta[0],
                    value=val,
                )
                for meta, val in zip(
                    itertools.product(metric, e_or_p, weighted, tagmod), Ps + Rs + Fs
                )
            ]
        )
    outf.close()


if __name__ == "__main__":
    Fire(main)
