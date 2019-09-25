# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask,render_template,request
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

import base64
import Model

import io
import matplotlib.pyplot as plt




 #mysql = MySQL(app)


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/prediction')
def prediction():
    return render_template('SS.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/register')
def register():
    return render_template('register.html')
@app.route   ('/results', methods = ['POST'])
def getValue():
    ca  = int(request.form['ca'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    thal=int(request.form['thal'])
    exang=int(request.form['exang'])
    oldpeak=float(request.form['oldpeak'])   
    thalach=int(request.form['thalach'])
    algo = request.form['algo']
    X=[[ca,cp,exang,oldpeak,sex,thal,thalach]]
    target_predicted,f_score,confusion_matrix,auc_plot,y_pred,data_for_plotting = Model.train_model(X,algo)
    
    fpr, tpr, thresholds = roc_curve(data_for_plotting, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.001, 1])
    plt.ylim([0, 1.001])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    img=io.BytesIO()
    img.seek(0)
    plt.savefig(img, format='png')
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    response ='data:image/png;base64,{}'.format(plot_url)
    return render_template('result.html',ip=X,t=target_predicted,algo=algo,f_score=f_score,confusion_matrix=confusion_matrix,auc_plot=auc_plot,response=response)
if __name__ == '__main__':
    app.run(debug=True)


