from flask import Flask, render_template, request
import joblib
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load trained model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        text = request.form["user_text"]

        # Convert text to features
        features = vectorizer.transform([text])

        # Prediction
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        human_conf = round(prob[0] * 100, 2)
        ai_conf = round(prob[1] * 100, 2)

        prediction = "AI-Generated Text" if pred == 1 else "Human-Written Text"

        # ✅ ALWAYS SAME COLORS ORDER
        # values = [AI, Human]
        colors = ["red", "green"]

        # Highlight predicted part
        if pred == 1:      # AI predicted
            explode = (0.12, 0)
        else:              # Human predicted
            explode = (0, 0.12)

        # Create Pie Chart
        fig, ax = plt.subplots(figsize=(4,4))

        ax.pie(
            [ai_conf, human_conf],
            labels=["AI Text", "Human Text"],
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            explode=explode
        )

        ax.set_title("AI vs Human Confidence", color="white")

        # Dark navy background
        fig.patch.set_facecolor("#001f3f")
        ax.set_facecolor("#001f3f")

        # Convert chart → image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    facecolor="#001f3f")
        buf.seek(0)
        chart_data = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        return render_template(
            "result.html",
            prediction=prediction,
            ai_conf=ai_conf,
            human_conf=human_conf,
            chart_data=chart_data
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
