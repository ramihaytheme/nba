from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import sys
app = Flask(__name__)
# Define the class LogTransformer applied for skewness
class LogTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(X)
    
# Load model and scaler
log_transformer = LogTransformer()
scaler = joblib.load('scaler.pkl')
model = joblib.load('logistic_regression_model.joblib')
list_of_features = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA',
       '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
       'TOV']
list_of_features_after = ['GP', 'MIN', 'PTS', 'FGM', 'FGA', 'FG%', '3P Made', '3PA',
       '3P%', 'FTM', 'FTA', 'FT%', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK',
       'TOV','FG_P','FT_P','TP_P']
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_features = np.array(data['input_features']).reshape(1, -1)
    input_dataframe = pd.DataFrame(input_features, columns=list_of_features)
    # Apply scaling
    transformed_features = log_transformer.transform(input_dataframe)
    transformed_features["FG_P"] = (transformed_features["FGM"] / transformed_features["FGA"])*100
    transformed_features["FT_P"] = (transformed_features["FTM"] / transformed_features["FTA"])*100
    transformed_features["TP_P"] = (transformed_features["3P Made"] / transformed_features["3PA"])*100
    scaled_features = pd.DataFrame(scaler.transform(transformed_features),columns=list_of_features_after)
    scaled_features = scaled_features.drop(columns=["REB","FTA","3P Made","3P%",'FGM',"MIN","STL","FT%"])
    # Get the prediction
    prediction = model.predict(scaled_features)
    
    # Send the response
    return jsonify(prediction=int(prediction[0]))


# To deploy first run the app.py file and then run the following command in the terminal:
# curl -X POST -H "Content-Type: application/json"      -d '{"input_features": [f_1, f_2, ... ]}'      http://localhost:5000/predict

if __name__ == '__main__':
    app.run(debug=True)
