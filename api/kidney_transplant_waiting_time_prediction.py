from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__, static_url_path='', static_folder='./static')
categorical_columns = ['gender', 'blood_gp', 'age_cat', 'cPRA_cat', 'gestation', 'prior_transplant', 'underlying_disease']
prediction_columns = ['age_at_list_registration', 'gender', 'dialysis_duration', 'blood_gp', 'age_cat', 'cPRA', 'HLA_A1', 'HLA_A2', 'HLA_B1',
                      'HLA_B2', 'HLA_DR1', 'HLA_DR2', 'cPRA_cat', 'gestation', 'prior_transplant', 'underlying_disease', 
                      'number_prior_transplant', 'if_transplanted', 'duration', 'log_time_on_Dialysis']
classification_columns = ['gender', 'underlying_disease', 'blood_gp', 'age_cat', 'cPRA_cat']
models = ['rf', 'xgb']

label_encoders = {}
classification_models = {}
model = pickle.load(open('static/models/coxph.pkl', 'rb'))
scaler = pickle.load(open('static/models/scaler.pkl', 'rb'))

def loadModels():
    for cols in categorical_columns:
        label_encoders[cols] = pickle.load(open(f'static/models/{cols}_label_encoder.pkl', 'rb'))
    for model in models:
        classification_models[model] = pickle.load(open(f'static/models/{model}_classifier.pkl', 'rb'))

def process_input_args(input_data):
    input_df = pd.DataFrame(input_data)
    input_subset_df = input_df[prediction_columns]
    
    for col in categorical_columns:
        input_subset_df[col] = label_encoders[col].transform(input_subset_df[col])

    data = pd.DataFrame(scaler.transform(input_subset_df))
    data.columns = input_subset_df.columns

    return data

def predict_transplantation_possibility(input_parameters):
    input_parameters = [input_parameters[classification_columns]]
    classification_predictions = {}
    for model in models:
        classification_predictions[model] = classification_models[model].predict(input_parameters)
    
    return classification_predictions


def predict_waiting_time(input_parameters):
    predicted_partial_hazard = model.predict_partial_hazard(input_parameters)
    predicted_survival_prob = np.exp(-predicted_partial_hazard)
    expected_transplant_time = -np.log(predicted_survival_prob)
    expected_transplant_time_array = expected_transplant_time.to_numpy()

    return [time for time in expected_transplant_time_array]



@app.route('/', methods=['GET'])
def start():
    return 'The service is up and running', 200

@app.route('/getTransplantPrediction', methods=['GET'])
def predict_transplantation():
    if request.args:
        args = request.args.to_dict() 
        input = { parameter: [(int(args[parameter]) if args[parameter].isnumeric() else args[parameter])] for parameter in prediction_columns }
        processed_input = process_input_args(input)
        prediction_list = list(processed_input.apply(lambda row: predict_transplantation_possibility(row), axis=1))
        print(prediction_list)
        
        return jsonify({model: prediction_list[0][model].tolist()[0] for model in models }), 200
    else:
        return 'Please give input parameters', 200

@app.route('/getWaitTime', methods=['GET'])
def predict_wait_time():
    if request.args:
        args = request.args.to_dict() 
        input = { parameter: [(int(args[parameter]) if args[parameter].isnumeric() else args[parameter])] for parameter in prediction_columns }
        processed_input = process_input_args(input)
        wait_time_list = list(processed_input.apply(lambda row: predict_waiting_time(row), axis=1))
        return jsonify({"Approximate wait time": str(round(wait_time_list[0][0], 2))+" Months" }), 200
    else:
        return 'Please give input parameters', 200

if __name__ == '__main__': 
    loadModels()
    app.run(debug=True)