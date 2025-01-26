import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
data = pd.read_csv('disease_outbreak_data.csv')

# Clean column names by stripping extra spaces
data.columns = data.columns.str.strip()

# Convert 'date' column to datetime format
data['date'] = pd.to_datetime(data['date'], errors='coerce')

# Add time-based features (month, day of week)
data['month'] = data['date'].dt.month
data['day_of_week'] = data['date'].dt.dayofweek

# Apply one-hot encoding for categorical variables (e.g., 'location')
data = pd.get_dummies(data, drop_first=True)

# Feature Selection
X = data.drop(['outbreak_level', 'date'], axis=1)  # Drop 'date' column after feature extraction
y = data['outbreak_level']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Classification Report
report = classification_report(y_test, y_pred, target_names=['No Outbreak', 'Outbreak'])

# Display Results in a User-Friendly Way
print("\nModel Evaluation Results:")
print("="*30)
print(f"Accuracy of the Model: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)

# Save the trained model
joblib.dump(model, 'disease_outbreak_model.pkl')
