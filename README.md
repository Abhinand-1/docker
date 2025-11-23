# News Sentiment Classifier (FastAPI + Docker)

Run locally:
1. pip install -r requirements.txt
2. python train_model.py
3. docker build -t news-sentiment-app .
4. docker run -p 8000:8000 news-sentiment-app
