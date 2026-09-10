# ===============================
# AI vs Human Text Predictor
# ===============================

import joblib
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load trained model & vectorizer
# -------------------------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print(" Model and Vectorizer loaded successfully!")
except:
    print(" Model or Vectorizer not found! first run train_model.py.")
    exit()

# -------------------------------
# 2️⃣ User Input
# -------------------------------
user_text = input("\nEnter text to analyze:\n")

# -------------------------------
# 3️⃣ Text to features
# -------------------------------
user_vector = vectorizer.transform([user_text])

# -------------------------------
# 4️⃣ Predict
# -------------------------------
prediction = model.predict(user_vector)[0]
probability = model.predict_proba(user_vector)[0]

human_confidence = probability[0] * 100
ai_confidence = probability[1] * 100

# -------------------------------
# 5️⃣ Show Result
# -------------------------------
print("\n========== RESULT ==========")
if prediction == 1:
    print("🧠 Prediction: AI-GENERATED TEXT")
else:
    print("🧠 Prediction: HUMAN-WRITTEN TEXT")

print(f"AI Confidence: {ai_confidence:.2f}%")
print(f"Human Confidence: {human_confidence:.2f}%")

# -------------------------------
# 6️⃣ Pie Chart
# -------------------------------
plt.figure(figsize=(6,6))
plt.pie(
    [ai_confidence, human_confidence],
    labels=["AI Text", "Human Text"],
    autopct="%1.1f%%",
    startangle=90,
    colors=["#ff9999", "#66b3ff"],
    explode=(0.1, 0)
)
plt.title("AI vs Human Text Confidence")
plt.show()




