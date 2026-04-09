# Tatar Detoxifier (Rule-Based)

Deterministic rule-based detoxification pipeline for **Tatar text moderation** in a low-resource setting.

## Why This Project
Low-resource NLP often requires a different mindset. When annotated data, pretrained resources, and strong benchmark baselines are limited, a transparent rule-based system can be the right first step.

This repository demonstrates that approach for Tatar text detoxification.

## Problem Statement
Reduce toxic or unsafe fragments in Tatar-language text using a deterministic transformation pipeline.

## Approach
The project combines:
- substring-based toxic pattern detection
- replacement dictionaries
- lexicon support for filtering and normalization
- deterministic transformation rules

The main value here is not model complexity. It is **controllability, transparency, and fast iteration** in a low-resource setting.

## Repository Structure
```text
main.py
paper.pdf
tat_Cyrl_twl.txt
toxic_replacements.json
toxic_substrings.json
tt_ru_lexicon.csv
README.md
```

## Why Rule-Based Was a Reasonable Choice
For this task, rule-based logic offers several benefits:
- explainable transformations
- no dependency on large labeled datasets
- easy debugging of failure cases
- good baseline for future supervised or hybrid systems

## Evaluation Mindset
A project like this should be reviewed through:
- coverage of toxic patterns
- correctness of replacements
- preservation of non-toxic content
- interpretability of the transformation pipeline

Useful reporting may include:
- examples of successful detoxification
- classes of remaining failure cases
- precision / recall on a manually curated evaluation set

## Running Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
python main.py
```

## Where This Project Is Strong
- low-resource NLP framing
- pragmatic baseline construction
- explicit lexical resources and rules
- easy discussion of trade-offs in moderation systems

## Suggested Next Improvements
- add a small annotated evaluation set
- classify toxic span categories explicitly
- include before/after examples in the README
- compare rule-based behavior against a lightweight learned baseline
- package the transformation logic as a reusable module or API

## Limitations
- rule-based systems need ongoing maintenance
- linguistic variation can reduce coverage
- aggressive substitutions may alter meaning if policy is not tuned carefully

## Takeaway
This repository is valuable because it reflects a real NLP engineering lesson: in low-resource settings, a strong transparent baseline is often more useful than pretending a complex model is justified too early.
