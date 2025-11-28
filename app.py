from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
# Make sure 'LinearRegressionModel.pkl' is in the same folder!
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, years=years, fuel_types=fuel_types)

@app.route('/models/<company>')
def load_models(company):
    models = car[car['company'] == company]['name'].unique()
    return jsonify({'models': list(models)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Debug Print: Check if request reaches here
        print("\n=== RECEIVED PREDICTION REQUEST ===")
        
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kilo_driven'))

        print(f"Inputs: {company} | {car_model} | {year} | {fuel_type} | {kms_driven}")

        # Create DataFrame
        input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]], 
                                  columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        
        # Predict
        prediction_inr = model.predict(input_data)[0]
        
        # Convert to PKR and Format
        exchange_rate = 3.36
        prediction_pkr = int(np.round(prediction_inr * exchange_rate, 0))
        
        # Format with commas (e.g., "PKR 1,500,000")
        formatted_price = f"PKR {prediction_pkr:,}"
        
        print(f"Prediction Success: {formatted_price}")
        
        # Return JSON success response
        return jsonify({'status': 'success', 'price': formatted_price})
    
    except Exception as e:
        # Print the REAL error to the terminal
        print(f"!!! ERROR DURING PREDICTION !!! : {str(e)}")
        # Return JSON error response
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)