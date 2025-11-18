# Customer Churn Prediction System

## Project Description  
This project predicts the likelihood of a customer churning (leaving) a telecom service using a real-world dataset and machine learning. It features data preprocessing, model training, and a graphical dashboard for predictions and insights.

## Problem Statement  
In subscription-based services like telecom, retaining customers is crucial. By identifying customers at risk of churning, companies can take proactive steps to retain them and reduce revenue loss.

## Technologies Used  
- Python 3.x  
- Pandas, NumPy   
- Streamlit (Interactive Dashboard)  
- sklearn
- joblib

## Project Structure  


customer_churn_project/
│── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│── model/
│ └── churn_model.pkl
│── src/
│ ├── data_preprocessing.py
│ ├── train_model.py
│ ├── predict.py
│── app/
│ └── dashboard.py
├── README.md
└── requirements.txt


## Installation & Setup  

:-  pip install -r requirements.txt


:-  (Optional) Train the model.
    cd src
    python train_model.py

:-   Run the Dashboard.
     cd app
     streamlit run dashboard.py

## Usage

Launch the dashboard.

View raw dataset by toggling the sidebar option.

Explore charts (churn distribution, tenure vs charges etc.).

Enter customer details in the prediction form and get result.

“Customer Will CHURN” or “Customer Will NOT Churn”.


## Features

Real-world dataset analysis and model training.

Interactive dashboard with charts and prediction form.

Extensible code structure for adding more features.


## Future Scope

Include more input features (InternetService, Contract type, PaymentMethod).

Add advanced visualisations (heatmaps, segmentation).

Deployment to cloud (Heroku, AWS) for live predictions.

Experiment with improved models (XGBoost, Neural Networks).

## Author / Contact

Lokesh Solanki
B.Tech IT Student – Arya College of Engineering, Jaipur
Email: lokesh.solanki@gmail.com