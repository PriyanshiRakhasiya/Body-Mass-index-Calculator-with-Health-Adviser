import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and encoders
MODEL_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'bmi_health_model.pkl')
ENCODER_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'bmi_encoder.pkl')
GENDER_ENCODER_PATH = os.path.join(app.config['UPLOAD_FOLDER'], 'gender_encoder.pkl')

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
gender_encoder = joblib.load(GENDER_ENCODER_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
      
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        age = int(request.form['age'])
        gender = request.form['gender']
        gender_encoded = gender_encoder.transform([gender])[0]  # Encode gender

        # Calculate BMI and predict category
        bmi = weight / (height ** 2)
        category_encoded = model.predict([[height, weight, age, gender_encoded]])[0]
        category = encoder.inverse_transform([category_encoded])[0]

 
        advice_df = pd.read_csv('uploads/BMI_Health_Advice.csv')
        advice_row = advice_df[(advice_df['Category'] == category) & (advice_df['Gender'] == gender)]
        
        advice = advice_row.iloc[0]['Advice'] if not advice_row.empty else "No specific advice available."

        return render_template('results.html', bmi=round(bmi, 2), category=category, advice=advice, age=age, gender=gender)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
