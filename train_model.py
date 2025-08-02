import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df[["hours_studied", "attendance_percent", "previous_score"]]
y = df["marks_obtained"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

