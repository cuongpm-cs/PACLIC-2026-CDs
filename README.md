# Cognitive Distortion Deep Learning Sources

This folder is a standalone package containing three Python training scripts and the two CSV files they need. The scripts use the same stratified 80/20 train/test split (`random_state=42`) so their held-out rows are consistent. They download pretrained DistilBERT checkpoints from Hugging Face on first use. Model weights are not included in this package.

## Contents

| File | Task | Input | Saved model |
| --- | --- | --- | --- |
| `distortion_detection.py` | Binary detection: distortion or no distortion | Full `Patient Question` | `artifacts/detection-transformer/` |
| `distortion_mrc_extraction.py` | Extract the distorted text span with question answering | Full question and a positive span annotation | `artifacts/span-extractor/` |
| `distortion_classification.py` | Classify one of 10 distortion types | Annotated distorted span | `artifacts/type-transformer/` |
| `data/distortion.csv` | Source annotations for detection | 2,530 questions | — |
| `data/distortion_processed.csv` | Source annotations with span fields for MRC and type classification | The same 2,530 questions | — |

`requirements.txt` is the standard dependency file. `requirement.txt` is an alias for the same file, included for compatibility with the requested filename. The scripts have no project-local imports from outside this folder.

## Install

Use Python 3.11 or 3.12. Open a terminal in this folder.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Training is faster with a GPU. If you need a specific CUDA or ROCm build of PyTorch, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). An internet connection is required to install dependencies and download the model checkpoints the first time.

## Validate the bundled data

These commands check paths and annotations without downloading a model or starting training:

```bash
python distortion_detection.py --check-data
python distortion_mrc_extraction.py --check-data
python distortion_classification.py --check-data
```

## Train and evaluate

```bash
python distortion_detection.py --epochs 2 --batch-size 8
python distortion_mrc_extraction.py --epochs 2 --batch-size 8
python distortion_classification.py --epochs 2 --batch-size 8
```

Each script saves its model and tokenizer under `artifacts/` and prints held-out evaluation results. Run the scripts independently. On a CPU, full training can take a long time; reduce the batch size if memory is limited. You can change the input and output locations with `--data` and `--output-dir`. Run `python <script>.py --help` for all options. Paths are resolved relative to the script by default, so the commands also work when launched from another directory.

## Evaluation notes

- Detection uses full questions and reports binary precision, recall, and F1.
- MRC trains only on positive examples. Some `Answer Start` values in the CSV are inconsistent; the script recomputes answer positions and skips spans absent from the question. It reports normalized exact match and token F1 on held-out positive examples.
- Type classification uses **gold annotated spans** at test time. Its classification report is an oracle-input result, not a complete detect–extract–classify pipeline score.
- The scripts split rows before filtering MRC or classification examples, preventing training on their held-out rows.

This code is for research and educational use, not clinical assessment.
