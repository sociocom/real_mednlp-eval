# Real-MedNLP CR & RR Eval

The evaluation toolkit for one of the NTCIR-16's evaluation tasks, [Real-MedNLP CR & RR tracks](https://sociocom.naist.jp/real-mednlp/).

## Setup

Please install dependencies listed in `pyproject.toml` by using Python package managers (e.g. `poetry install`).

## Usage

This toolkit provides two functionalities: validate the XML format and calculate evaluation metrics.

### Format validation

Use `validate_format.py`:

```
python validate_format.py --ref <reference XML file> --dtd <DTD file> path/to/your_submission.xml
```

- `--ref`: path to the reference XML file (i.e. the test file without annotation: `MedTxt-CR-JA_or_EN-Test.xml`)
- `--dtd`: path to the XML DTD file (`MedTxt.dtd`)

This script checks three points for the submission XML file:

- XML syntax: whether it is valid as an XML file
- XML scheme: whether it follows the given DTD file
- Bare-text match: whether its plain text (tag-removed text) matches to the original test file

### Metric calculation

#### Subtask 1, 2: NER

Use `evaluate_ner.py`:

```
python evaluate_ner.py --ref <test_addTag.xml> --tagset cr --attrib path/to/your_submission.xml
```

The script outputs Precision, Recall, F-score, and Support for each tag as well as micro/macro averaged scores thereof.

- `--ref`: path to the reference XML file with annotated (`MedTxt-CR-JA_or_EN-Test_addTag.xml`)
- `--tagset`: tagset to evaluate (all, cr, rr)
  - `all`: include all tags defined in PRISM annotation except `<p>`
  - `cr`: target "d", "a", "timex3", "t-test", "t-key", "t-val", "m-key", and "m-val"
  - `rr`: evaluate "d", "a", "timex3", and "t-test"
- `--attrib` (`--no-attrib`): consider (or ignore) the attributes defined in some tags ("certainty", "state", "type")

To calculate the finer metrics, use `finer_ner_eval.py`:

```
python finer_ner_eval.py </dir/path/to/IOB> --resume False
```

- The finer metrics include:
  - Partial match scores of Precision, Recall, and F1-score
  - Training frequency-based weighting to Precision, Recall, and F1-score
- The script reads IOB files from the given directory and write the scores CSV file (all metrics of one system per line) to the current directory
  - IOB files can be converted from the submission XML format by using `evaluate_ner.py`'s `convert_xml_to_iob()` function
- `--resume`: some systems may generate errorneous IOB files. If you want to resume the evaluation from the last successful file, set this option to `True`.

#### Subtask 3: ADE

Use `evaluate_ade.py`:

```
python evaluate_ade.py --ref <test_answer.csv> path/to/your_submission.csv
```

The script outputs Precision, Recall, F-score, and Support for each ADEval (0--3) as well as micro/macro averaged scores thereof.

- `--ref`: path to the test answer CSV (`MedTxt-CR-JA_or_EN-ADE-test_answer-v2.csv`)



#### Subtask 3: CI

Use `evaluate_ci.py`:

```
python evaluate_ci.py --ref <test_answer.csv> path/to/your_submission.csv
```

The script outputs Normalized Mutual Information score.

- `--ref`: path to the test answer CSV (`MedTxt-CR-JA_or_EN-CI-test_answer.csv`)
