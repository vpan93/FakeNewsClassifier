import logging
import os
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template
import warnings
from pydantic import BaseModel, ValidationError
from typing import List
from logging.handlers import RotatingFileHandler

# Import your FakeNewsPredictor module methods
from FakeNewsPredictor import preprocessing, train_algorithm, predict, tune_hyperparameters

# Ignore all warnings
warnings.filterwarnings("ignore")

class Statement(BaseModel):
    statement: str
    subjects: str
    speaker_name: str
    speaker_job: str
    speaker_state: str
    speaker_affiliation: str
    statement_context: str

# Initialize Flask app
app = Flask(__name__)

# Check and delete the existing app.log file
if os.path.exists('app.log'):
    os.remove('app.log')

# Setup logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    app.logger.info("Received file upload request")

    if 'csvfile' not in request.files or 'txtfile' not in request.files:
        app.logger.error('Missing file in request')
        return jsonify(error='Missing file'), 400

    csv_file = request.files['csvfile']
    txt_file = request.files['txtfile']

    if csv_file.filename == '' or txt_file.filename == '':
        app.logger.error('No selected file')
        return jsonify(error='No selected file'), 400

    if csv_file and txt_file:
        df = pd.read_csv(csv_file)
        
        try:
            file_content = txt_file.read().decode('utf-8')
            statements = json.loads(file_content)
            json_data = [Statement(**statement).dict() for statement in statements]
        except ValidationError as e:
            app.logger.error(f"Error in JSON structure: {e}")
            return jsonify(error=f"Error in JSON structure: {e}"), 400
        except json.JSONDecodeError as e:
            app.logger.error(f"Error parsing JSON: {e}")
            return jsonify(error=f"Error parsing JSON: {e}"), 400

        app.logger.info("Starting data preprocessing..")
        X_train, X_val, X_test, y_train, y_val, y_test, tokenizer, max_sequence_length = preprocessing(df)

        app.logger.info("Starting hyperparameter tuning process..")
        best_hyperparams = tune_hyperparameters(X_train, y_train, X_val, y_val, tokenizer, max_sequence_length)
    
        app.logger.info("Training model..")
        model, optimizer = train_algorithm(tokenizer, max_sequence_length, X_train, y_train, X_val, y_val, best_hyperparams)

        from joblib import dump
        app.logger.info("Saving model..")

        # Assuming 'model' is your trained model
        dump(model, 'FakeNewsClassifierPackage/model/latest_trained_model.joblib')

        app.logger.info("Predicting..")
        predictions_with_labels = predict(json_data, model, tokenizer, max_sequence_length)


        # Save predictions to a .txt file
        results_dir = os.path.join(os.path.dirname(__file__), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(os.path.join(results_dir, 'predictions.txt'), 'w') as f:
            json.dump(predictions_with_labels, f, indent=4)

        app.logger.info("Uploaded files processed successfully!")
        return jsonify(predictions_with_labels)

@app.route('/get-logs')
def get_logs():
    with open('app.log', 'r') as file:
        return file.read()

if __name__ == "__main__":
    app.run(debug=True)
