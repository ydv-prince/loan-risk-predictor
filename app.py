from flask import Flask, render_template, request, send_from_directory
import json
import os
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

# Load top features for rendering form
with open("artifacts/top_features.json") as f:
    feature_info = json.load(f)
top_features = feature_info["top_features"]

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        user_input = {}
        for f in top_features:
            val = request.form.get(f)
            if val is not None and val.strip() != "":
                try:
                    user_input[f] = float(val.strip())
                except ValueError:
                    pass

        pipeline = PredictPipeline()
        prediction = pipeline.predict_from_dict(user_input)
    return render_template('form.html', top_features=top_features, prediction=prediction)

@app.route('/batch_predict', methods=['GET', 'POST'])
def batch_predict():
    output_path = None
    if request.method == 'POST':
        file = request.files['file']
        upload_dir = os.path.join("artifacts", "prediction_artifacts")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)
        pipeline = PredictPipeline()
        output_path = pipeline.predict_from_csv(file_path)
    return render_template('upload.html', output_path=output_path)

@app.route('/download/<filename>')
def download_file(filename):
    predictions_dir = os.path.join("artifacts", "predictions")
    return send_from_directory(predictions_dir, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)