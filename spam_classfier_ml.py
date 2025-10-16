import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

data = {
    'text': [
        'Win money now!', 
        'Hello, how are you?', 
        'Limited offer, buy now!', 
        'Meeting at 10am', 
        'Congratulations, you won a prize!', 
        'Lunch tomorrow?'
    ],
    'label': [1, 0, 1, 0, 1, 0] 
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.3, random_state=42, stratify=df['label']
)


vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))


def predict_spam(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)
    return 'Spam' if pred[0] == 1 else 'Not Spam'

print(predict_spam("Free money for you!"))

