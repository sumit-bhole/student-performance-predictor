import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df[["hours_studied", "attendance_percent", "previous_score"]]
y = df["marks_obtained"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save model using pickle
with open("RF_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)

