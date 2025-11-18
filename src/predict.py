import joblib
import pandas as pd
from data_preprocessing import preprocess_data


data = {
    "customerID": ["0001-ABCD"],     
    "gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [5],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [70],
    "TotalCharges": [300],
    "Churn": ["No"]     
}

df = pd.DataFrame(data)
df.to_csv("single_input.csv", index=False)

X, y = preprocess_data("single_input.csv")


model = joblib.load("../model/churn_model.pkl")


pred = model.predict(X)

if pred[0] == 1:
    print("\nPrediction: Customer Will CHURN ❌")
else:
    print("\nPrediction: Customer Will NOT Churn ✅")
