# SummarEase
### Making Long Reads Effortless
This project was created for EECS 487: Natural Language Processing.

**SummarEase** is a hybrid summarization tool that combines extractive (TextRank) and abstractive (PEGASUS) models to generate concise, high-quality summaries of long-form content such as news articles, research papers, and biomedical abstracts.

---

## Features
- **Extractive Summarization** using Sumy’s TextRank
- **Abstractive Summarization** using:
  - `google/pegasus-xsum` (short summaries)
  - `google/pegasus-cnn_dailymail` (longer summaries)
- **Hybrid Summarization**: Extractive summary is passed through a fine-tuned PEGASUS model
- **Evaluation Metrics**:
  - ROUGE-1, ROUGE-2, ROUGE-L
  - BERTScore (semantic similarity)
  - Flesch Reading Ease & SMOG Index
  - Compression Ratio
  - Repetition Score
 
---

## Sample Workflow in `runall.py`
- Clean and standardize the input article
- Generate:
  - Extractive summary
  - Abstractive summaries (XSum, CNN/DM)
  - Hybrid summary
- Evaluate each with all metrics

---

## Installation

```bash
python3.11 -m venv summar-ease-env
source summar-ease-env/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

Ensure Python 3.11 is installed via Homebrew if needed:
```bash
brew install python@3.11
```

---

## How to Run
```bash
python runall.py
```

This will output all summaries and metrics for the built-in climate change article.

---

## Authors
- Naeem Saleem  
- Reeva Faisal  
- Merjem Memic  
- Eshan Chishti



