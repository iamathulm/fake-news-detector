import joblib
from src.preprocess import clean_text

model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_news(text):

    text = clean_text(text)

    vector = vectorizer.transform([text])

    prediction = model.predict(vector)[0]

    if prediction == 1:
        return "True News"
    else:
        return "Fake News"