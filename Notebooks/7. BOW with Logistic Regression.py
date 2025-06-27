import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- Load the Data ---
# Update these paths as needed
train_csv = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\dataset_splits\train\train.csv"
test_csv = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\dataset_splits\test\test.csv"

# Load CSV files
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Rename columns if necessary (assuming original columns 'v2' for text and 'v1' for label)
train_df = train_df.rename(columns={'sms': 'text', 'label': 'label'})
test_df = test_df.rename(columns={'sms': 'text', 'label': 'label'})

print("Training samples:", len(train_df))
print("Test samples:", len(test_df))

# Extract text and labels
X_train, y_train = train_df['text'], train_df['label']
X_test, y_test = test_df['text'], test_df['label']

# --- BOW Baseline with TF-IDF and Logistic Regression ---
# Create TF-IDF vectorizer and transform data
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression classifier
clf = LogisticRegression(max_iter=1000, random_state=69420)
clf.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = clf.predict(X_test_tfidf)

# Evaluate performance
print("Logistic Regression Baseline Performance:")
print(classification_report(y_test, y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test, y_pred))

# --- Simple Baselines ---

# 1. Random Baseline
# Compute the random classifier accuracy over many iterations
n_iterations = 10000
random_accuracies = []
for _ in range(n_iterations):
    random_preds = np.random.choice([0, 1], size=len(y_test))
    random_accuracies.append(accuracy_score(y_test, random_preds))
random_baseline_accuracy = np.mean(random_accuracies)
print("\nRandom Baseline (empirical) Average Accuracy:", random_baseline_accuracy)

# 2. Majority-Class Baseline
# Determine the most frequent class in the training set
majority_class = Counter(y_train).most_common(1)[0][0]
majority_preds = [majority_class] * len(y_test)
print("\nMajority-Class Baseline Performance:")
print(classification_report(y_test, majority_preds, digits=4))
print("Accuracy:", accuracy_score(y_test, majority_preds))
