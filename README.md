# Real-MedNLP CR & RR Eval

The evaluation toolkit for Real-MedNLP (NTCIR-16)'s CR & RR tracks.

## Requirement

You need the following external packages to run this toolkit.

- `lxml`
- `seqeval`
- `scikit-learn`

Please install these packages to your Python environment, e.g.:

```
pip install --user lxml seqeval scikit-learn
```

The commands would be different depending on your Python setup.

Also, you can create a virtual environment with `poetry` by using the given `pyproject.toml`.

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
