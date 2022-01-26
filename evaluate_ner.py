"""Evaluate NER results."""
import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from seqeval.metrics import classification_report

TAGSET = {
    "all": [
        "d",
        "a",
        "c",
        "f",
        "timex3",
        "t-test",
        "t-key",
        "t-val",
        "m-key",
        "m-val",
        "r",
        "cc",
    ],  # exclude "p"
    "cr": ["d", "a", "timex3", "t-test", "t-key", "t-val", "m-key", "m-val"],
    "rr": ["d", "a", "timex3", "t-test"],
}


def tagname_with_attr(elem):
    for key in ["state", "certainty", "type"]:
        if key in elem.attrib:
            return f"{elem.tag}_{elem.attrib[key][:3]}"
    return elem.tag


def convert_xml_to_iob(root, tagset="all", attrib=False):
    # tagset: all, cr, rr
    x = []  # for debug
    y = []
    for article in root.findall(".//article"):
        chrs = []
        lbls = []
        for elem in article.iter():
            # this iter starts from <article> itself
            if elem.text:
                text = elem.text.lstrip("\n") if elem.tag == "article" else elem.text
                chrs.extend(list(text))
                if elem.tag in TAGSET[tagset]:
                    tagname = tagname_with_attr(elem) if attrib else elem.tag
                    lbls.extend([f"B-{tagname}"] + [f"I-{tagname}"] * (len(text) - 1))
                else:  # elem.tag == "article":
                    lbls.extend(["O"] * len(text))
            if elem.tail and elem.tag != "article":
                # <article>'s tail is outside of the article -> skip
                chrs.extend(list(elem.tail))
                lbls.extend(["O"] * len(elem.tail))
                # print(chrs, lbls)
        assert len(chrs) == len(lbls), article.attrib

        sent_x = []  # for debug
        sent_y = []
        for char, label in zip(chrs, lbls):
            if char != "\n":
                sent_x.append(char)
                sent_y.append(label)
            else:  # if '\n', drop the char and create a sentence
                x.append(sent_x)
                y.append(sent_y)
                sent_x = []
                sent_y = []

    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate NER performance for Real-MedNLP Subtask 1 and 2."
    )
    parser.add_argument("xml", type=Path, help="path to the submission XML")
    parser.add_argument(
        "--ref",
        type=Path,
        help="path to the reference (test_addTag) XML",
        required=True,
    )
    parser.add_argument(
        "--tagset", type=str, help="tagset to evaluate (all, cr, rr)", default="all"
    )
    parser.add_argument(
        "--attrib",
        action=argparse.BooleanOptionalAction,
        help="whether or not consider tag attributes (certainty, state, type)",
        default=False,
    )
    args = parser.parse_args()

    root = ET.parse(args.ref)
    x_true, y_true = convert_xml_to_iob(root, tagset=args.tagset, attrib=args.attrib)

    root = ET.parse(args.xml)
    x_pred, y_pred = convert_xml_to_iob(root, tagset=args.tagset, attrib=args.attrib)

    if x_true != x_pred:
        print(
            "Input texts do not match. Make sure you validate the XML.", file=sys.stderr
        )
        for i, (t, p) in enumerate(zip(x_true, x_pred)):
            if t != p:
                print("line", i)
                print("--", "".join(t))
                print("++", "".join(p))
        exit(1)

    print(classification_report(y_true, y_pred))

    # with open("evaluate_interim.tsv", "w") as f:
    #     for sent_x, sent_y in zip(x_pred, y_pred):
    #         for x, y in zip(sent_x, sent_y):
    #             f.write(f"{x}\t{y}\n")
    #         f.write("\n")
