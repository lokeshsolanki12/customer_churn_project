from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import preprocess_data


X, y = preprocess_data("../data/WA_Fn-UseC_-Telco-Customer-Churn.csv")


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, Y_train)


pred = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, pred))


joblib.dump(model, "../model/churn_model.pkl")

print("Model saved successfully!")
