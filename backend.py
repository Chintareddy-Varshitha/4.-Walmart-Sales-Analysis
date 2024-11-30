from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your model
model_path = r"C:\Users\Yasho Nandan Reddy\Desktop\ml_backend\random_forest_model.pkl"  # Use raw string
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Route to get data
@app.route('/get_data', methods=['GET'])
def get_data():
    try:
        # Load the recent data from CSV
        csv_path = r"C:\Users\Yasho Nandan Reddy\Downloads\WALMART_SALES_DATA.csv"  # Use raw string
        data = pd.read_csv(csv_path)

        # Get the last 7 weeks of data
        recent_data = data.tail(7)

        # Prepare features for prediction
        features = recent_data[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']]
        predictions = model.predict(features)

        # Prepare data for chart
        result = recent_data[['Date', 'Weekly_Sales']].copy()
        result['Predicted_Sales'] = predictions

        # Format the result for the chart
        chart_data = result.rename(columns={'Date': 'week', 'Predicted_Sales': 'value'}).to_dict(orient='records')

        return jsonify(chart_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Return error message

if __name__ == '__main__':
    app.run(debug=True)
