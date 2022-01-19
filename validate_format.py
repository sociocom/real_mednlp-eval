"""Validate XML format. """
import argparse
import sys
import tempfile
import xml.etree.ElementTree as ET
from difflib import unified_diff
from pathlib import Path

from lxml import etree


def get_xml_validated(fp, dtdp):
    # check parsing
    try:
        root = ET.parse(fp)
    except ET.ParseError as e:
        print("[NG] XML syntax", file=sys.stderr)
        print("---------------", file=sys.stderr)
        print(e, file=sys.stderr)
        with open(fp, "r") as f:
            line, col = e.position
            error_line = f.readlines()[line - 1]
            # print(error_line)
            start_ix = max(0, col - 20)
            end_ix = min(col + 20, len(error_line) - 1)
            focus = error_line[start_ix:end_ix]
            if start_ix != 0:
                focus = "..." + focus
            if end_ix != len(error_line) - 1:
                focus += "..."
            print(focus, "|", repr(error_line[col]), file=sys.stderr)
        exit(1)
    print("[OK] XML syntax")

    # check scheme
    dtd = etree.DTD(dtdp.name)
    root_lxml = etree.parse(str(fp)).getroot()
    if not dtd.validate(root_lxml):
        print("[NG] XML scheme", file=sys.stderr)
        print("---------------", file=sys.stderr)
        for dtd_err in dtd.error_log.filter_from_errors():
            print(dtd_err)
        exit(1)
    print("[OK] XML scheme")

    return root


def remove_tags(root):
    for article in root.findall(".//article"):
        new_text = "".join(list(article.itertext()))
        attr = article.attrib.copy()
        article.clear()
        article.text = new_text
        article.attrib = attr
        article.tail = "\n"


def check_diff(ref, tgt):
    # d = Differ()
    # return d.compare(ref, tgt)
    return list(unified_diff(ref, tgt, fromfile="REF", tofile="TGT"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove annotation tags from articles")
    parser.add_argument("xml", type=Path, help="path to your submission XML")
    parser.add_argument(
        "--ref", type=Path, help="path to the reference (test) XML to check diff"
    )
    parser.add_argument(
        "--dtd",
        type=Path,
        help="path to the DTD of the reference XML",
    )
    args = parser.parse_args()

    root = get_xml_validated(args.xml, args.dtd)

    remove_tags(root)
    fto = tempfile.TemporaryFile()
    root.write(fto, encoding="utf-8")

    # check bare text
    with open(args.ref, "r") as fref:
        ref_xml = fref.readlines()
    fto.seek(0)
    tgt_xml = [l.decode("utf-8") for l in fto]
    diff_lst = check_diff(ref_xml, tgt_xml)
    if diff_lst:
        print("[NG] Bare text", file=sys.stderr)
        print("--------------", file=sys.stderr)
        for d in diff_lst:
            print(d.rstrip(), file=sys.stderr)
        exit(1)
    else:
        print("[OK] Bare text")
    print("[OK] All checkpoints")
    exit(0)
