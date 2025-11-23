import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = fetch_20newsgroups(subset="all", categories=["rec.autos","sci.electronics"], remove=("headers","footers","quotes"))
texts = data.data
labels = ["negative" if t==0 else "positive" for t in data.target]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))

joblib.dump(pipe, "model.pkl")
