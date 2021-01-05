import pandas as pd 
import pickle
import yaml
import numpy as np 
from flask import Flask, request, url_for, redirect, render_template, jsonify
import base64
import random
from feature_engg import model_ready_data

app = Flask(__name__)

lime_data = pd.read_csv('assets/PersonalLoan_processed_data.csv').drop(columns = ['PersonalLoan'])

global id

def randN():
    N=7
    min = pow(10, N-1)
    max = pow(10, N) - 1
    id = random.randint(min, max)
    return id

@app.route('/', methods = ['GET'])
def home():
    global id
    id = randN()
    file = 'home.html'
    return render_template(file, id=id)

@app.route('/PersonalLoan', methods = ['POST']) 
def get_data():
    
    explain_data = request.data.decode()
    
    explain_data = explain_data.split(',')
    print(request.data)
    predict_data = []
    for val in explain_data:
#         if val.isdigit() == True:
        predict_data.append(val)
#         elif val.replace('.', '', 1).replace('"','',1).isdigit() == True:
#             predict_data.append(float(val))
#         else:
#             predict_data.append(str(val))
            
    file = open('assets/PersonalLoan_model.pkl','rb')
    model = pickle.load(file)
    file.close()
    
    data = pd.DataFrame(columns = ['Age', 'Experience', 'Income', 'ZIPCode', 'Family', 'CCAvg', 'Education', 'Mortgage', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard'])
    data.loc[0,:] = predict_data
    model_data = model_ready_data(data)
    
    print(model_data)
        
    return_data = {}
    
    prediction = model.predict(np.array(model_data.iloc[0,:]).reshape(1,-1))
    
    # return_data['prediction'] = str(np.expm1(prediction))
    
    # return_data = jsonify(return_data)
    
    file = 'home.html'
    return render_template(file, id=id, pred=str(np.expm1(prediction)))

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))
#     app.run()
