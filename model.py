import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor

# Load and preprocess data
df = pd.read_csv('employee_data.csv')

label_encoders = {}
for col in ['JobTitle', 'Education', 'Location']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('Salary', axis=1)
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train ensemble models
rf = RandomForestRegressor(random_state=42)
gb = GradientBoostingRegressor(random_state=42)
vr = VotingRegressor(estimators=[('rf', rf), ('gb', gb)])

vr.fit(X_train, y_train)

# Prediction function
def predict_salary(job, experience, education, location):
    input_data = {
        'JobTitle': label_encoders['JobTitle'].transform([job])[0],
        'Experience': experience,
        'Education': label_encoders['Education'].transform([education])[0],
        'Location': label_encoders['Location'].transform([location])[0]
    }
    input_df = pd.DataFrame([input_data])
    prediction = vr.predict(input_df)[0]
    return prediction
