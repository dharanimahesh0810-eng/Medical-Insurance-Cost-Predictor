"""
train_model.py
----------------
Trains the Insurance Cost Prediction model (Linear Regression) and saves
the artifacts needed by app.py (the Streamlit interface).

Run this ONCE before starting the Streamlit app:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load data -------------------------------------------------------------
# Make sure insurance.csv is in the same folder as this script
df = pd.read_csv("insurance.csv")

# 2. Encode categorical columns, keeping the fitted encoders so the
#    Streamlit app can apply the exact same encoding to new inputs
encoders = {}
for col in ["sex", "smoker", "region"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Split features / target ------------------------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model --------------------------------------------------------
model = make_pipeline(LinearRegression())
model.fit(x_train, y_train)

# 5. Evaluate -------------------------------------------------------------
prediction = model.predict(x_test)
print("MAE:", mean_absolute_error(y_test, prediction))
print("MSE:", mean_squared_error(y_test, prediction))
print("R2:", r2_score(y_test, prediction))

# 6. Save artifacts needed by the Streamlit app ------------------------
joblib.dump(model, "insurance_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("Saved: insurance_model.pkl, encoders.pkl, feature_columns.pkl")
