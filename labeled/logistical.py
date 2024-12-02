import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Train the model for multi-class classification
logistic_reg = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')  # Use multinomial for multi-class
logistic_reg.fit(X_train, y_train)

# Make predictions
y_train_pred = logistic_reg.predict(X_train)  # Training predictions
y_test_pred = logistic_reg.predict(X_test)    # Testing predictions
y_val_pred = logistic_reg.predict(X_val)      # Validation predictions

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
validation_accuracy = accuracy_score(y_val, y_val_pred)

# Print accuracies
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")
print(f"Validation Accuracy: {validation_accuracy:.2f}")

# Print Confusion Matrix and Classification Report for Validation Data
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
report_val = classification_report(y_val, y_val_pred)

print("\nConfusion Matrix (Validation):")
print(conf_matrix_val)

print("\nClassification Report (Validation):")
print(report_val)




