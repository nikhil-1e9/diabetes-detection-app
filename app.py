import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("diabetes-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    feature_list = request.form.to_dict()
    feature_list = list(feature_list.values())
    feature_list = list(map(int, feature_list))
    final_features = np.array(feature_list).reshape(1, 14)   
    prediction = model.predict(final_features)

    if int(prediction[0]) == 0:
        output = "You are safe"
    else:
        output = "You are at high risk of getting diabetes"

    return render_template(
        "index.html", prediction_text=output
    )

if __name__ == "__main__":
    app.run(debug=True)
