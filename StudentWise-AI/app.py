from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model_filename = "ridge_regression_model1.pkl"
best_model_pipeline = joblib.load(model_filename)


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction Page
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    # When opening the page normally
    if request.method == "GET":
        return render_template("predict.html")

    # Get values safely from the form
    gender = request.form.get("gender")
    race = request.form.get("race")
    education = request.form.get("parental_education")
    lunch = request.form.get("lunch")
    prep_course = request.form.get("test_prep")
    math_score = request.form.get("math_score")
    reading_score = request.form.get("reading_score")

    # Check whether all values were received
    if not all([
        gender,
        race,
        education,
        lunch,
        prep_course,
        math_score,
        reading_score
    ]):
        return render_template(
            "predict.html",
            error="Please fill in all the fields before submitting."
        )

    # Convert scores to numbers
    math_score = float(math_score)
    reading_score = float(reading_score)

    # Create input DataFrame
    new_data = pd.DataFrame({
        "gender": [gender],
        "race/ethnicity": [race],
        "parental level of education": [education],
        "lunch": [lunch],
        "test preparation course": [prep_course],
        "math score": [math_score],
        "reading score": [reading_score]
    })

    # Predict writing score
    predicted_score = best_model_pipeline.predict(new_data)

    # Round prediction
    rounded_score = round(predicted_score[0], 2)

    # Show result
    return render_template(
    "result.html",
    prediction=rounded_score,
    math_score=math_score,
    reading_score=reading_score
)


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)