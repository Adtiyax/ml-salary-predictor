import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# -----------------------------
# Step 1: Load and clean data
# -----------------------------
df = pd.read_csv('employee_data.csv')

# Select relevant columns
df = df[['job_title', 'experience_level', 'employee_residence', 'company_location', 'salary_in_usd']]

# -----------------------------
# Step 2: Encode categorical features
# -----------------------------
label_encoders = {}
for col in ['job_title', 'experience_level', 'employee_residence', 'company_location']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# Step 3: Split the data
# -----------------------------
X = df.drop('salary_in_usd', axis=1)
y = df['salary_in_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 4: Train ensemble models
# -----------------------------
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)
vr = VotingRegressor(estimators=[('rf', rf), ('gb', gb)])

vr.fit(X_train, y_train)

# -----------------------------
# Step 5: Save model and encoders
# -----------------------------
with open('salary_model.pkl', 'wb') as f:
    pickle.dump(vr, f)

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ Model and encoders saved successfully.")
