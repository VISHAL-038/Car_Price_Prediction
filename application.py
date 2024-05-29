#flask,pandas , scikit-learn, pickle-mixin

from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
car=pd.read_csv("Cleaned_Car_data.csv")
model = pickle.load(open("LinearRegressionModel.pkl","rb"))

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = sorted(car['fuel_type'].unique())
    companies.insert(0,"Select Company")
    return render_template('index.html',companies=companies,car_models=car_models,years=year,fuel_type=fuel_type)

# predict route
@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kilo_driven'))
    # print(company,car_model,year,fuel_type,kms_driven)
    prediction = model.predict(pd.DataFrame([[car_model,company,year,fuel_type,kms_driven]],columns=['name','company','year','fuel_type','kms_driven']))
    # print(company,car_model)
    # print(prediction[0])
    return str(np.round(prediction[0],2))
    
if __name__ == "__main__":
    app.run(debug=True)

