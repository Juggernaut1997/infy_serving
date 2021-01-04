import pandas as pd 
import pickle
import yaml
import numpy as np 
from flask import Flask, jsonify, request
import base64
from feature_engg import model_ready_data

app = Flask(__name__)

lime_data = pd.read_csv('assets/PersonalLoan_processed_data.csv').drop(columns = ['PersonalLoan'])

@app.route('/PersonalLoan', methods = ['POST']) 
def get_data():
    
    explain_data = request.data.decode()
    
    explain_data = explain_data.split(',')
    print(request.data)
    predict_data = []
    for val in explain_data:
        if val.isdigit() == True:
            predict_data.append(int(val))
        elif val.replace('.', '', 1).replace('"','',1).isdigit() == True:
            predict_data.append(float(val))
        else:
            predict_data.append(str(val))
            
    file = open('assets/PersonalLoan_model.pkl','rb')
    model = pickle.load(file)
    file.close()
    
    data = pd.DataFrame(columns = ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education', 'Mortgage', 'Securities Account', 'CD Account', 'Online', 'CreditCard'])
    data.loc[0,:] = predict_data
    model_data = model_ready_data(data)
    
    print(model_data)
        
    return_data = {}
    
    prediction = model.predict(np.array(model_data.iloc[0,:]).reshape(1,-1))
    
    return_data['prediction'] = str(np.expm1(prediction))
    
    return_data = jsonify(return_data)
    
    return return_data

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
#     app.run()
        
        
    
    
    
    
    
    
    
    
    
    
    
