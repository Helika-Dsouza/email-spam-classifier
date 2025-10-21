import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Tiny toy dataset — replace with your real labeled dataset (CSV, database, etc.)
data = {
    "text": [
        "Win money now!",
        "Hello, how are you?",
        "Limited offer, buy now!",
        "Meeting at 10am",
        "Congratulations, you won a prize!",
        "Lunch tomorrow?",
        "Claim your free gift today!",
        "Are we still on for dinner?",
        "Exclusive deal just for you!",
        "Project deadline is Friday",
        "You’ve been selected for a reward!",
        "Can you send me the report?"
    ],
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}
df = pd.DataFrame(data)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
)

# Build a pipeline so preprocessing + model stay together
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),  # try uni- and bi-grams
            ),
        ),
        ("clf", MultinomialNB()),
    ]
)

# Fit
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Optional: cross-validated predictions (better estimate on small data)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_preds = cross_val_predict(pipeline, df["text"], df["label"], cv=cv)
print("Cross-validated classification report:\n", classification_report(df["label"], cv_preds, zero_division=0))

# Save the fitted pipeline for later use
joblib.dump(pipeline, "spam_pipeline.joblib")

def predict_spam(text):
    if not isinstance(text, str) or text.strip() == "":
        return "Not Spam"  # or raise ValueError, depending on your needs
    pred = pipeline.predict([text])
    return "Spam" if int(pred[0]) == 1 else "Not Spam"

# Example
print(predict_spam("Free money for you!"))
