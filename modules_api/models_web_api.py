# Dependencies
from os import system
from flask import Flask, request, jsonify

import traceback
import pandas as pd
import numpy as np
from prediction_api import *

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if randomForest:
        try:
            json_ = request.json
            print(json_)
            query = pd.DataFrame(json_)           
            X_transformed = preprocessing(query)
            y_pred = randomForest.predict(X_transformed)
            y_proba = randomForest.predict_proba(X_transformed)
            
            return jsonify({'prediction': y_pred,'prediction_proba':y_proba[0][0]})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Problem loading the model')
        return ('No model here to use')


if __name__ == '__main__':
    try:
        port = int(system.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    randomForest = pickle.load(open("classifier_rf_model.sav", 'rb'))
    print ('Models loaded')

    app.run(port=port, debug=True)