from flask import Flask, render_template, request, redirect, url_for, flash
from flask import Flask, render_template, session
from flask_mysqldb import MySQL
import MySQLdb
from flaskext.mysql import MySQL
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import sys
import pickle
import numpy as np
import plotly
import plotly.graph_objs as go
import json
import plotly.express as px


app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
columns=['Unixtime','date','hour','radiation','temperature','pressure','humidity',
'winddirection','speed','sunrise','sunset']
df = pd.read_csv('SolarPrediction.csv',header=0,names=columns)

@app.route('/')
def homepage():
    return render_template("index.html")

@app.route('/predict')
def prediction():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict(): 

    float_vals=[float(x) for x in request.form.values()]
    vals=[np.array(float_vals)]
    prediction = model.predict(vals)
    output = round(prediction[0], 2)
    return render_template('predict.html', prediction='The Prediction of Solar Radiation  is: {}'.format(output),plot_prediction='static/assets/img/prediction.png')

@app.route('/eda.html', methods=['GET','POST'])
def eda():

    bar = create_plot()
    return render_template('eda.html', plot=bar,  plot_url1='/static/assets/img/boxplot.png',plot_url2='/static/assets/img/heatmap.png')

def create_plot():
    fig=px.scatter(df, x="temperature", y="radiation",size="radiation", color="radiation", hover_name="temperature",size_max=20)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route('/tsa.html', methods=['GET','POST'])
def tsa():   
    return render_template('tsa.html',  tsa_plot1='/static/assets/img/radiation.png',tsa_plot2='/static/assets/img/temperature.png',
    tsa_plot3='/static/assets/img/humidity.png',tsa_plot4='/static/assets/img/pressure.png',tsa_plot5='/static/assets/img/winddirection.png',
    tsa_plot6='/static/assets/img/speed.png')
    
if __name__ == "__main__":
    app.run()