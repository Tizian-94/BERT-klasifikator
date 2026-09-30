# BERT Klasifikator

Python BERT-based AI classifier for **Croatian Accounting (HNB) code sequences**.

## What it does

Classifies sequences of Croatian accounting codes using a BERT-based NLP model. Useful for automated categorization of accounting entries, compliance checking, or financial text analysis in Croatian.

## Tech

| Component | Technology |
|---|---|
| Model | BERT (HuggingFace transformers) |
| Language | Python |
| Task | Sequence classification — Croatian accounting codes (HNB) |
| Data | Custom accounting code sequences |

## Project Structure

```
BERT-klasifikator/
├── BERT-klasifikator/   # Main package
├── config/              # Configuration
├── data/                # Training/eval data
├── env/                 # Python virtualenv
├── models/              # Saved model artifacts
├── notebooks/           # Exploration & experiments
├── LICENSE
└── README.md
```

## Related

Part of a broader exploration of NLP + fintech applications — combining transformer models with domain-specific financial text in Croatian.

---

Built as an experiment in applying transformer-based NLP to accounting code classification.
