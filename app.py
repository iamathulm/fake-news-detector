from src.predict import predict_news

print("Fake News Detector")

while True:

    text = input("\nEnter news text (type 'exit' to quit): ")

    if text.lower() == "exit":
        break

    result = predict_news(text)

    print("Prediction:", result)