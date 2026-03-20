# tatar-detoxifier-rule-based

Educational rule-based detoxification pipeline for Tatar text.

## Repository Contents

- `main.py` - main detoxification script.
- `toxic_replacements.json` - token replacement dictionary.
- `toxic_substrings.json` - substring-based filtering rules.
- `tat_Cyrl_twl.txt` - additional lexicon list.
- `tt_ru_lexicon.csv` - lexicon source data.

## Implemented Functionality

The code performs deterministic text detoxification using:

- exact-token replacement rules,
- substring filtering for toxic patterns,
- dictionary-based cleanup from external lexicon files,
- output generation in tabular format.
