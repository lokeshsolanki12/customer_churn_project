import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(path):
    df = pd.read_csv(path)

 
    df = df.dropna()

   
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return X, y
