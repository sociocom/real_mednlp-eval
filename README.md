# Real-MedNLP CR Eval

The evaluation toolkit for Real-MedNLP (NTCIR-16)'s CR track.

## Requirement

You need the following external packages to run this toolkit.

- `lxml`
<!-- - `seqeval` -->

Please install these packages to your Python environment, e.g.:

```
pip install --user lxml
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

- `--ref`: path to the reference XML file (`MedTxt-CR-JA_or_EN-Test.xml`)
- `--dtd`: path to the XML DTD file (`MedTxt.dtd`)

This script checks three points for the submission XML file:

- XML syntax: whether it is valid as an XML file
- XML scheme: whether it follows the given DTD file
- Bare-text match: whether its plain text (tag-removed text) matches to the original test file

### Metric calculation

TBA
