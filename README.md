# 📱 SMS Spam Detection with NLP

This repository contains a full pipeline for detecting SMS spam using Natural Language Processing techniques. Built as part of a graduate-level NLP course project, this work explores multiple classification approaches ranging from classical methods like Logistic Regression to fine-tuned transformer models like BERT and ELECTRA.

---

## 📦 Dataset

The dataset consists of **5,575 SMS messages**, labeled as either **Spam (1)** or **Not Spam (0)**:
- **Spam**: 748 entries (~15.5%)
- **Not Spam**: 4,827 entries (~84.5%)

The data was preprocessed and split into:
- **Training**: 60% (3,344 samples)
- **Validation**: 20% (1,115 samples)
- **Test**: 20% (1,115 samples)

---

## 💡 Methods & Models

### 1. **Logistic Regression with TF-IDF (BOW Baseline)**
- Simple and fast to implement.
- Used `TfidfVectorizer` (10,000 features, stopword removal).
- **Accuracy**: 95.87%
- **Spam F1 Score**: 0.823

### 2. **Fine-Tuned Transformers (Hugging Face)**

#### 🔹 BERT (`bert-base-uncased`)
- Fine-tuned using Hugging Face `transformers`.
- **Test Accuracy**: 99.37%
- **Spam F1 Score**: 0.976

#### 🔹 ELECTRA (`google/electra-small-discriminator`)
- Surprisingly high performance despite smaller parameter count.
- **Test Accuracy**: 98.92%
- **Spam F1 Score**: 0.959

### 3. **Zero-Shot Classification**
- Tested with:
  - `facebook/bart-large-mnli`
  - `valhalla/distilbart-mnli-12-3`
- Explored different prompts (e.g., `"This text is {}"`).
- **Performance was significantly lower** than fine-tuned models.
- **Spam F1 Score (Best Case)**: ~0.205

---

## 📊 Results Summary

| Model                          | Accuracy | Spam Precision | Spam Recall | Spam F1 Score |
|-------------------------------|----------|----------------|--------------|----------------|
| BERT (Fine-Tuned)             | 99.37%   | 97.31%         | 96.67%       | 0.970          |
| ELECTRA (Fine-Tuned)          | 99.01%   | 97.28%         | 95.33%       | 0.963          |
| Logistic Regression (TF-IDF)  | 95.87%   | 96.40%         | 71.81%       | 0.823          |
| Zero-Shot (DistilBART)        | 34–83%   | 9–12%          | 4–71%        | ~0.20          |
| Majority Class Baseline       | 86.6%    | 0.0%           | 0.0%         | 0.0            |
| Random Baseline               | ~50%     | –              | –            | –              |

---

## 📁 Project Structure

