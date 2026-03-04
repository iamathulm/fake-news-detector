import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.preprocess import clean_text

print("Loading dataset...")

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

print("Cleaning text...")

data["text"] = data["text"].apply(clean_text)

X = data["text"]
y = data["label"]

print("Vectorizing text...")

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

model = LogisticRegression()

model.fit(X_train, y_train)

print("Evaluating model...")

pred = model.predict(X_test)

print(classification_report(y_test, pred))

print("Saving model...")

joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Training complete!")