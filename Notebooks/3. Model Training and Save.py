import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
from tqdm import tqdm

# File paths for your dataset splits
train_csv = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\dataset_splits\train\train.csv"
val_csv = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\dataset_splits\val\val.csv"
test_csv = r"C:\Users\dalla\Desktop\Natural Language Processing\Homework 3\dataset_splits\test\test.csv"

# Load the datasets
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# Display dataset sizes
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

# Extract text and labels; assumes columns are named 'text' and 'label'
X_train, y_train = train_df['sms'], train_df['label']
X_val, y_val = val_df['sms'], val_df['label']
X_test, y_test = test_df['sms'], test_df['label']

# Create TF-IDF vectorizer and transform the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Train logistic regression on the training set
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_tfidf, y_train)

# Evaluate on the validation set
val_preds = clf.predict(X_val_tfidf)
print("Validation Set Performance:")
print(classification_report(y_val, val_preds, digits=4))
print("Validation Accuracy:", accuracy_score(y_val, val_preds))

# Evaluate on the test set
test_preds = clf.predict(X_test_tfidf)
print("\nTest Set Performance:")
print(classification_report(y_test, test_preds, digits=4))
print("Test Accuracy:", accuracy_score(y_test, test_preds))

# Save the trained model and vectorizer for future use
joblib.dump(clf, 'logistic_regression_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
print("\nModel and vectorizer saved successfully.")
