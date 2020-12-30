import pandas as pd 
import numpy as np
import pickle
import sys

def load_data(data_path):
    data = pd.read_csv(data_path)
    return data


def model_ready_data(data):
    file = open('assets/training.pkl','rb')
    model = pickle.load(file)
    file.close()
    
    data['LoanPredict'] = model.predict(data.drop('PersonalLoan', axis=1))
    
    return data

def save_data(data):
    data.to_csv('assets/PersonalLoan_processed_data.csv',index = False)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == 'training':
        data = load_data('datasets/PersonalLoan_treated_data.csv')
        data = model_ready_data(data)
        save_data(data)
    elif mode == 'serving':
        data = model_ready_data()
    
    
    


        
