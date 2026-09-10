import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

print("📂 Dataset loading...")
df = pd.read_csv("AI_Human.csv")

# Balance dataset
human_df = df[df['generated']==0.0].sample(n=181438, random_state=42)
ai_df = df[df['generated']==1.0]
balanced_df = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42)

print("Dataset balanced")

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(balanced_df["text"])
y = balanced_df["generated"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# Save
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("💾 Model & Vectorizer saved")
