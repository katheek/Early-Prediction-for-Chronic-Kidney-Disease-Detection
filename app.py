from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open('CKD.pkl', 'rb'))
le_dict = pickle.load(open('label_encoders.pkl', 'rb'))
le_target = pickle.load(open('target_encoder.pkl', 'rb'))

# Expected input order
selected_features = [
    'specific_gravity',
    'albumin',
    'blood_glucose_random',
    'serum_creatinine',
    'haemoglobin',
    'packed_cell_volume',
    'red_blood_cell_count',
    'hypertension'
]

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_dict = {}
        for feature in selected_features:
            input_dict[feature] = request.form[feature]

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Convert numeric fields first
        for col in input_df.columns:
            if col not in le_dict:
                input_df[col] = input_df[col].astype(float)

        # Encode categorical fields
        for col in input_df.columns:
            if col in le_dict:
                input_df[col] = le_dict[col].transform(input_df[col])

        input_df = input_df[selected_features]  # Ensure order

        # Prediction
        pred = model.predict(input_df)
        predicted_label = le_target.inverse_transform(pred)[0]

        return render_template('result.html', prediction_text=predicted_label)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=False)

