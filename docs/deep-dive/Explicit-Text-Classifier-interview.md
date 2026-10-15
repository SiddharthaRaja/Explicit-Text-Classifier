# Explicit-Text-Classifier — Interview Questions & Model Answers

> Companion to [`Explicit-Text-Classifier.md`](./Explicit-Text-Classifier.md). First-person, 30–90 s answers. 🟢 Easy · 🟡 Medium · 🔴 Hard

**If you only prep 10:** A1, A3, C3, D1, D3, E1, E2, G1, H1, and all hostile follow-ups (X1 especially).

---

## (a) Project overview & motivation

**A1. 🟢 Describe the project.**
> It's a classical NLP baseline for explicit-content detection. I load about 1,200 labelled articles from HuggingFace, de-duplicate to 1,072, clean and lemmatise them, build TF-IDF features over unigrams and bigrams, and train a linear SVM. On a 20% held-out split it gets 86% accuracy. On the explicit class it has 0.87 precision but only 0.62 recall, and improving that recall is the main next step.

**A2. 🟢 Why this problem?**
> Content moderation is a high-volume, high-stakes classification problem with asymmetric costs. Missing explicit content is usually worse than a false alarm. It's a good setting for thinking about thresholds and metrics beyond accuracy.

**A3. 🟡 What are your results?**
> Accuracy 0.86, macro-F1 0.82. For the explicit class: precision 0.87, recall 0.62, F1 0.73. For non-explicit: precision 0.86, recall 0.96. I want to flag that an earlier README version reported 0.86/0.96 as the explicit class's numbers. That came from a label-ordering mistake in `classification_report`, which I've since diagnosed.

**A4. 🟡 Why start with TF-IDF + SVM instead of a transformer?**
> It's the right baseline: fast, interpretable (you can read the top-weighted n-grams), and strong on small text datasets. It tells you how much a transformer actually buys you. With about a thousand documents, a linear model is also less likely to overfit than a fine-tuned 66M-parameter model, though DistilBERT is the obvious next comparison.

**A5. 🟢 What would you do differently?**
> Start with the evaluation design: a stratified CV pipeline, a recall target, and a labelled challenge set of hard negatives. Then build models to that spec.

---

## (b) Architecture & design

**B1. 🟢 Walk through the pipeline.**
> Load the HF dataset to pandas, map labels to 1/0, drop duplicate articles, preprocess (lowercase, regex to letters only, spaCy lemmas, NLTK stopword removal), TF-IDF with (1,2)-grams, an 80/20 split, `SVC(kernel='linear', C=100)`, a classification report, and a `detect_explicit` function for single strings.

**B2. 🟡 Why lemmatise? Is it necessary with TF-IDF?**
> It merges inflections (killed/killing → kill), which reduces sparsity on a small dataset. But it's not always beneficial. Bigrams plus IDF often work just as well without it, and spaCy lemmatisation is the slowest step. I'd A/B it in CV, and if I keep it, disable the parser and NER components.

**B3. 🟡 Why bigrams?**
> Phrases carry meaning single words don't, like "sexual assault" or "graphic violence". The cost is a much bigger vocabulary. With `min_df=1` and about a thousand documents, most bigrams appear once, which adds noise. I'd tune `min_df` and `max_features`.

**B4. 🟡 How would you design the "needs review" tier?**
> Calibrate the SVM scores (Platt scaling via `CalibratedClassifierCV`), then pick two thresholds on validation data. The upper one gives high precision for auto-flagging. The lower one is set to hit a recall target like 95%, and everything between goes to human review. You then report the volume going to review as a cost. The README describes this, but the notebook doesn't implement it yet.

**B5. 🔴 How would you productionise it?**
> Wrap preprocessing, vectoriser and model in one sklearn `Pipeline`, serialise it with joblib, and serve it behind a FastAPI endpoint returning label, calibrated score and tier. Log inputs and decisions for auditing. Monitor drift through the score distribution and review rates. Retrain periodically on reviewed cases, which is effectively active learning.

---

## (c) Code-level / implementation

**C1. 🟡 Walk me through `preprocess`.**
> Lowercase, then regex `[^a-z\s]` removes everything except letters and whitespace. Then spaCy parses it, and I keep `token.lemma_` for tokens whose text isn't a stopword or punctuation. Two issues. The regex runs before spaCy, so contractions break ("didn't" → "didnt") and spaCy's lemmatiser sees unnatural text. And NLTK's stopword list includes "not" and "no", so negation is lost.

**C2. 🟢 Why `drop_duplicates` before splitting?**
> Otherwise the same article can land in both train and test, which inflates scores. It removed 117 exact duplicates (1,189 → 1,072). Near-duplicates, like syndicated news with small edits, aren't caught.

**C3. 🔴 Explain the `target_names` bug.**
> `classification_report` orders rows by sorted label: 0, then 1. `target_names` just renames them in that order. I passed `['positive','negative']`, so class 0 (Not Explicit) was printed as "positive" and class 1 (Explicit) as "negative". The supports give it away: 151 vs 64, while the data is 757 class-0 to 315 class-1. So the 0.96 recall belongs to the non-explicit class. The real explicit recall is 0.62.

**C4. 🟡 Is fitting TF-IDF on all data before splitting a problem?**
> It's mild leakage. The vocabulary and IDF weights include test documents, so test terms aren't "unseen", and IDF reflects the test distribution. The effect is usually small, but it's free to avoid: put the vectoriser in a Pipeline and fit inside each CV fold.

**C5. 🟡 `SVC(kernel='linear')` vs `LinearSVC`?**
> Same model family, different solvers. `SVC` uses libsvm's dual solver, which scales roughly quadratically to cubically with the number of samples. `LinearSVC` uses liblinear in the primal, which is much faster on large sparse text. `SVC` also doesn't give probabilities unless `probability=True`, and that runs an internal CV for Platt scaling.

**C6. 🟢 What does `detect_explicit` return?**
> In the final version, a string: "Explicit", "Not Explicit", or an empty-input warning. It doesn't return a score, so the README's `label, score = detect_explicit(text)` example would fail.

**C7. 🟡 The notebook execution counts are out of order. Why does that matter?**
> Saved outputs might depend on state from cells run in a different order, or on cells since edited. For example, cell 3's output shows `clean_text`, which cell 5 creates. For reproducibility you should restart and run all, or move the logic into a script.

---

## (d) Underlying ML/CS theory

**D1. 🟡 Explain TF-IDF.**
> Term frequency counts how often a term appears in a document. Inverse document frequency down-weights terms that appear in many documents. sklearn uses ln((1+n)/(1+df)) + 1. Multiply them and L2-normalise each document vector. Distinctive terms get high weight, and ubiquitous ones like "said" get low weight.

**D2. 🟡 Explain the SVM objective and what C does.**
> Minimise ½‖w‖² + C·Σ hinge loss, where hinge = max(0, 1 − y·f(x)). The first term maximises the margin, and the second penalises points on the wrong side or inside the margin. Large C means a smaller margin and fitting training points more tightly, i.e. less regularisation. C = 100 on about 860 docs with far more features than docs is close to a hard-margin SVM, which risks overfitting.

**D3. 🔴 Your data has p ≫ n. What does that imply?**
> With more features than samples, the training data is almost always linearly separable, so a linear SVM can reach zero training error for many C values. Generalisation then depends on the margin and on regularisation, so C, `min_df` and feature selection need tuning by CV. Simpler regularised models like logistic regression with L2 often generalise just as well.

**D4. 🟡 Why isn't `decision_function` a probability?**
> It's the signed distance w·x + b to the hyperplane. It ranks examples, but its scale is arbitrary and depends on C and the feature scaling. To threshold it meaningfully you either tune thresholds on validation data directly or calibrate it into probabilities (Platt scaling fits a sigmoid; isotonic regression fits a monotone step function).

**D5. 🟡 Why can't bag-of-words tell "discussing" from "being" explicit content?**
> It discards word order and context. The vector for "the platform banned explicit sexual content" overlaps heavily with explicit content itself. Contextual models like BERT encode each token conditioned on the full sentence, so they can learn mention-vs-use.

**D6. 🟢 Precision vs recall in moderation?**
> Recall: of all explicit items, what fraction we catch. Precision: of everything we flag, what fraction is actually explicit. Moderation usually prioritises recall, sometimes with a human-review tier to protect precision. F2 or recall-at-fixed-precision are good single numbers.

---

## (e) Evaluation & metrics

**E1. 🔴 Is your evaluation sound?**
> Partly. Exact duplicates are removed before splitting, which is good. But it's a single unstratified split with only 64 explicit test examples, so recall has a CI of about ±0.12. TF-IDF is fit on everything, there's no hyperparameter search or CV, accuracy is the headline even though a majority-class baseline gets 70%, and the report mislabelled the classes. I'd move to stratified 5-fold CV inside a Pipeline and report PR-AUC, F2 and recall at a fixed precision.

**E2. 🟡 What's the majority baseline and why does it matter?**
> Always predicting "Not Explicit" gives 151/215 ≈ 70% accuracy. So 86% is a real but modest gain. And accuracy hides that a third of explicit articles are missed.

**E3. 🟡 How would you choose an operating threshold?**
> On validation folds, compute the precision-recall curve from calibrated scores, pick the threshold that achieves the target recall (say 0.9), report the precision at that point, and then check it on the held-out test set. Ideally the target comes from business costs, i.e. the cost of a miss vs the cost of a review.

**E4. 🟡 What test sets would you add?**
> A hard-negative challenge set (policy documents, legal text, news about moderation) to measure false positives on meta-discussion. A set of short user-generated texts, because the training data is long news articles. And per-topic slices to catch spurious correlations, like crime news being labelled explicit.

**E5. 🟢 How do you get confidence intervals?**
> Bootstrap the test predictions (resample with replacement 1,000 times and take percentiles), or use Wilson intervals for proportions like recall. CV across folds also gives a spread.

---

## (f) Scaling, production & system design

**F1. 🟡 How would this scale to millions of documents per day?**
> TF-IDF + linear model inference is extremely cheap: a sparse dot product. The bottleneck is spaCy. I'd drop lemmatisation or use `nlp.pipe` with disabled components and multiprocessing. HashingVectorizer avoids storing a vocabulary and makes the transform stateless.

**F2. 🟡 How would you handle drift?**
> Monitor the score distribution and the flag rate, sample flagged and unflagged items for human audit, and retrain on newly reviewed data. Vocabulary drift (new slang) is a big problem for TF-IDF, which is another argument for subword models.

**F3. 🔴 Design a two-stage moderation system.**
> Stage 1: a cheap, high-recall model (this one, or a small transformer) on everything. Stage 2: a heavier model (a fine-tuned RoBERTa or an LLM judge) on stage-1 positives and borderline cases. Stage 3: human review for the uncertain band. Measure end-to-end recall, review volume and latency. Close the loop by feeding reviewed items back as training data.

**F4. 🟢 What artefacts are needed at inference?**
> The fitted vectoriser (vocabulary + IDF), the model, the exact preprocessing code and versions (spaCy model, NLTK stopword list), and the threshold(s). Bundle them in one serialised Pipeline to avoid training/serving skew.

---

## (g) Failure modes & debugging

**G1. 🔴 Your report says recall 0.96 for "positive" but your product misses lots of explicit articles. How do you debug?**
> Check the confusion matrix with explicit label ids, not names. Then check support counts against class frequencies: 151 vs 64 when the data is 70% class 0 means "positive" is class 0. Then `labels=[0,1]` with the correct `target_names`. Root cause: renamed rows. Lesson: always print `confusion_matrix(y_test, y_pred, labels=[0,1])` alongside the report.

**G2. 🟡 A moderation policy page gets flagged as explicit. Why?**
> It shares vocabulary with explicit text, and TF-IDF has no notion of context. Mitigations: hard-negative training examples, a contextual model, or a review tier for borderline scores.

**G3. 🟡 "She trembled as he ran his fingers…" is predicted Not Explicit. Why?**
> No strong explicit tokens survive, and the training data is long news articles where "explicit" seems to mean crime and violence coverage. A single literary sentence is out of distribution. It's also unclear the dataset's label even covers this kind of text.

**G4. 🟢 Re-running the notebook gives different numbers. Why?**
> Out-of-order execution and hidden state. `train_test_split` is seeded, but if dedupe or preprocessing ran differently, the rows change. Restart and run all, or convert to a script.

**G5. 🟡 Preprocessing is slow. Why?**
> spaCy's full pipeline (tagger, parser, NER) runs per article via `apply`. Use `nlp.pipe(texts, batch_size=…, n_process=…)` with `disable=["parser","ner"]`.

---

## (h) "Why didn't you…" / alternatives

**H1. 🟡 Why not use class weights?**
> I should have. `class_weight='balanced'` would trade some precision for higher explicit recall, which is what moderation needs. Combined with threshold tuning, it's the cheapest fix for the 0.62 recall.

**H2. 🟡 Why not logistic regression?**
> It's a great alternative: similar accuracy on text, natively probabilistic outputs for thresholding, and fast. I'd include it in the same CV comparison.

**H3. 🟡 Why not fine-tune a transformer?**
> It's the planned next step. The dataset is small, so I'd use DistilBERT or RoBERTa-base with class weights, a few epochs, and early stopping on CV folds, then compare on identical splits.

**H4. 🟢 Why remove stopwords?**
> It's habit, and it shrinks the vocabulary. But IDF already down-weights common words, and removing "not"/"no" hurts. I'd test with and without.

**H5. 🟡 Why `C=100`?**
> Honestly, it wasn't tuned. It should be chosen by CV, likely somewhere in 0.1–10 for normalised TF-IDF features.

---

## (i) Behavioural (STAR)

**I1. 🔴 Tell me about a mistake in your reported results.**
> **S:** My README reported 0.96 recall on the explicit class. **T:** While preparing to defend the project, I re-checked the evaluation. **A:** The support counts didn't match the class balance, so I traced it to `target_names` renaming sorted labels in the wrong order. The 0.96 was the non-explicit class. **R:** I corrected the numbers (explicit recall 0.62) and rewrote the narrative around improving recall. Now I always print a confusion matrix with explicit label ids. I'd much rather catch that myself than have someone else catch it.

**I2. 🟡 Tell me about choosing a baseline.**
> **S:** A small labelled text dataset. **T:** Get a credible first model quickly. **A:** I went with TF-IDF + linear SVM: fast to train, easy to inspect, and known to be strong on small text tasks. **R:** 86% accuracy in minutes, which gave a clear bar for any transformer to beat, and made the recall gap obvious.

**I3. 🟡 Tell me about working with an ambiguous label definition.**
> The dataset's "Explicit" class turned out to include crime and violence news, while my mental model was sexual content, which is what my test sentences checked. I'd now start by writing a labelling guideline, reading 50 random examples per class, and agreeing on the definition before modelling.

**I4. 🟢 Tell me about collaborating on this.**
> It was a four-person team project for BITS F464 Machine Learning, reproducing a paper on detecting inappropriate content in text. Per our report's work split, I generated the paper's results, designed the model and framework, and ran the experiments. Teammates proposed extensions (BERT, LIME/SHAP, adversarial robustness), verified results, and wrote up the report. The notebook outputs also show it ran on a teammate's machine. *(Adjust to what you actually did.)*

---

## Hostile follow-ups

**X1. 🔴 "Your README claims 0.96 recall on explicit content. Your notebook shows 0.62. Which is it?"**
> 0.62. The 0.96 is the non-explicit class. It was mislabelled because `target_names` renames rows in sorted label order, and I passed them in the wrong order. The supports, 151 and 64, prove which is which. I've corrected it.

**X2. 🔴 "The README says you used hard negatives and a 3-tier threshold. Where are they?"**
> They're not in the notebook. That README section describes the intended design, not the implementation. It shouldn't have been written in the past tense. Implementing the threshold tiers properly needs calibration and a validation set, which I'd add first.

**X3. 🔴 "C=100, no CV, single split, 64 positives. Isn't this just a lucky split?"**
> It might be. The recall CI is about ±0.12. I'd rerun with stratified 5-fold CV inside a Pipeline, tune C and class weights, and report the mean ± std. I expect explicit recall to move meaningfully with class weighting alone.

**X4. 🔴 "Your 'explicit' class is mostly crime news. Does your model detect explicit content at all?"**
> It detects whatever the dataset's annotators meant by "Explicit", which looks like sensitive or violent news. My ad-hoc sexual-content sentences test a different concept, so they're not a fair check. Defining the label precisely is step zero for a real moderation system.

**X5. 🟡 "Why is your stopword list removing 'not'?"**
> NLTK's English stopword list includes negations. For sentiment or moderation that's harmful: "not explicit" becomes "explicit". I'd use a custom list or skip stopword removal.

**X6. 🟡 "Who is Akhil? And your course report says F1 0.91. Which is right?"**
> *(Answer truthfully about Akhil.)* On the metric: the course report repeated the same label mix-up. 0.91 is the non-explicit class's F1, and the explicit class's F1 is 0.73. The report is in the wrong repo (E-Commerce-Marketplace). I'd move it here with a corrected results table.

**X7. 🔴 "Why should we believe you understand ML evaluation if your own project mislabelled its main metric?"**
> Because I found it, understood exactly why it happened, and changed my process: explicit label ids, a confusion matrix next to every report, and sanity checks of supports against class frequencies. Mistakes like this are common in practice. What matters is having checks that catch them.

**X8. 🟡 "What's the first thing you'd do with one more day?"**
> Pipeline + stratified CV + class weights + calibrated thresholds tuned for 90% recall. Then a hard-negative challenge set to measure false positives on meta-discussion.
