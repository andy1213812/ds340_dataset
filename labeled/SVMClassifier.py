import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the datasets
train_test_data = pd.read_csv('20240208_120000_lss.csv')
validation_data = pd.read_csv('20240208_120000_vld.csv')

# Data Preprocessing for training and testing
X = train_test_data.drop(columns=['risk_level'])  # Drop target column
y = train_test_data['risk_level']  # Target column

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Preprocessing for validation
X_val = validation_data.drop(columns=['risk_level'])  # Drop target column
y_val = validation_data['risk_level']  # Target column

# Support Vector Machine (SVM) Model
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_val)

# Evaluate the model
print(f"Validation Accuracy: {accuracy_score(y_val, svm_preds):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, svm_preds))
print("Classification Report:")
print(classification_report(y_val, svm_preds))
