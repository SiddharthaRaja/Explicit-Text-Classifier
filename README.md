# Explicit Content Classifier

A binary text classifier that detects explicit content in articles using a linear-kernel SVM with a full NLP preprocessing pipeline. Built with a focus on high recall to minimize missed detections in safety-critical filtering, and extended with confidence thresholding and hard-negative training to reduce false positives on meta-discussion text.

---

## Results

| Metric | Score |
|---|---|
| Precision (explicit class) | **0.86** |
| Recall (explicit class) | **0.96** |

---

## How It Works

**Preprocessing pipeline:**
1. Lowercase → regex clean (strip non-alpha)
2. spaCy lemmatization
3. NLTK stopword removal
4. TF-IDF vectorization with unigrams + bigrams (`ngram_range=(1,2)`)

**Model:** SVM with linear kernel, `C=100`

**Inference:** A `detect_explicit()` function returns one of three tiers based on the model's raw `decision_function` score:

| Score | Label |
|---|---|
| `> 0.8` | 🚩 EXPLICIT |
| `0.3 – 0.8` | ⚠️ NEEDS REVIEW |
| `< 0.3` | ✅ NOT EXPLICIT |

---

## Key Design Decisions

**High recall over precision** — the classifier is tuned to minimize missed detections. In content moderation, a false negative (missing explicit content) is more costly than a false positive.

**Hard negatives in training** — a known failure mode of TF-IDF classifiers is flagging text that *discusses* explicit content (moderation policy docs, academic NLP papers, legal briefs) as explicit itself. To counter this, the training set includes targeted hard negatives: sentences that contain explicit-adjacent vocabulary but are clearly not explicit.

**Confidence thresholding** — instead of a binary prediction, the classifier exposes a 3-tier output. The NEEDS REVIEW band surfaces borderline cases for human review rather than making a hard call.

---

## Project Structure

```
Explicit_Content_Classifier.ipynb   # Full pipeline: preprocessing → training → evaluation
README.md
```

---

## Setup & Usage

```bash
# 1. Create environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install pandas scikit-learn spacy nltk datasets matplotlib seaborn

# 3. Download NLP models
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords')"

# 4. Run notebook
jupyter notebook Explicit_Content_Classifier.ipynb
# Cell > Run All
```

> **Note:** The dataset (`valurank/Explicit_content`) downloads automatically via HuggingFace. No CSV required. Restricted/sandboxed networks without HuggingFace access must run locally.

### Test a custom input

```python
text = "your sentence here"
label, score = detect_explicit(text)
print(label, score)
```

---

## Stack

| Component | Detail |
|---|---|
| Dataset | `valurank/Explicit_content` via HuggingFace |
| Preprocessing | spaCy lemmatization + NLTK stopwords |
| Features | TF-IDF, unigrams + bigrams |
| Model | SVM, linear kernel, `C=100` |
| Language | Python 3 |

---

## Known Limitations & Planned Improvements

- [ ] **Sentence-level classification** — score each sentence independently and aggregate, rather than treating a full paragraph as one unit
- [ ] **Fine-tune DistilBERT** — transformer models understand context ("explicit content was flagged" ≠ "explicit content"), which would eliminate most remaining false positives on meta-discussion text
- [ ] **Expand hard negatives** — more coverage of news articles, legal documents, and platform policy pages
