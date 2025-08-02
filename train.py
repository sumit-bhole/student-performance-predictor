import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load your synthetic dataset
df = pd.read_csv("student_data.csv")

X = df[["hours_studied", "attendance_percent", "previous_score"]]
y = df["marks_obtained"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "RF_model.pkl")