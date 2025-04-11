from flask import Flask, render_template, request
import re
import pandas as pd
import pickle
import joblib

model = pickle.load(open('knn.pkl','rb'))
processed1 = joblib.load('processed1')
processed2 = joblib.load('processed2')
# connecting to sql by creating sqlachemy engine
from sqlalchemy import create_engine
engine = ("postgresql+psycopg2://{user}:{pw}@localhost/{db}"
                       .format(user = "postgres", # user
                               pw = "swap", # password
                               db = "wbcd_db")) # database


#define flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        wbcd = pd.read_excel(f)
        newprocessed1 = pd.DataFrame(processed1.transform(wbcd))
        newprocessed2 = pd.DataFrame(processed2.transform(newprocessed1))
        predictions = pd.DataFrame(model.predict(newprocessed2),columns = ['diagnosis'])
        final = pd.concat([predictions, wbcd], axis = 1)


        final.to_sql('wbcd_predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        
       
        return render_template("new.html", Y = final.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
