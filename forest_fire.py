#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load and preprocess the dataset
data = pd.read_csv("Forest_fire.csv")  # Load the CSV data into a Pandas DataFrame

# Convert DataFrame to NumPy array for further processing
data = np.array(data)

# Extract features (X) and target variable (y)
# Assuming the first column is not needed (likely an ID), and the last column is the target variable
X = data[1:, 1:-1]  # Features: all rows, from column 1 to second last column
y = data[1:, -1]    # Target: all rows, last column

# Convert data types to integer for model compatibility
y = y.astype('int')
X = X.astype('int')

# Split data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Initialize the Logistic Regression model
log_reg = LogisticRegression()

# Train the model on the training data
log_reg.fit(X_train, y_train)

# Example input for prediction (manual test with sample data)
input_features = [int(x) for x in "45 32 60".split(' ')]  # Example input: "45 32 60" for temperature, humidity, wind speed
final_input = [np.array(input_features)]  # Convert input to the correct format (array)

# Predict the probabilities of the input features
prediction = log_reg.predict_proba(final_input)

# Print predicted probabilities
print(f"Prediction probabilities: {prediction}")

# Save the trained model to a file for later use
pickle.dump(log_reg, open('model.pkl', 'wb'))

# Load the model to confirm it was saved correctly
model = pickle.load(open('model.pkl', 'rb'))

# Optionally, you can also test the loaded model to ensure it's working
test_prediction = model.predict_proba(final_input)
print(f"Test Prediction probabilities after loading the model: {test_prediction}")
