# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import xgboost as xgb
from flask import Flask,render_template,url_for,request

def word_divide_char(inputs):
    characters=[]
    for i in inputs:
        characters.append(i)
    return characters

clf = pickle.load(open('XGBPasswordstrength.pkl', 'rb'))
cv = pickle.load(open('tranformword.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if(request.method == 'POST'):
        message = request.form['message']
        data = np.array([message])
        vect = cv.transform(data)
        my_prediction = clf.predict(vect)
        my_prediction = int(my_prediction)
        
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
    



