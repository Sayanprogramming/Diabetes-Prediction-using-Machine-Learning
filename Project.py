import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score, classification_report

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

# Load the dataset from a local file
dataset = 'diabetes prediction dataset.csv'  
df = pd.read_csv(dataset)

# Visualize the first 5 rows of the dataset 
print(df.head()) 

# Check and print the shape of the DataFrame
print("Shape of the dataset:", df.shape) 

# Check for missing values
print(df.isnull().sum())  

# Define features (X) and target (y)
X = df.drop('Outcome', axis=1) 
y = df['Outcome']

# Standardize the features
scaler = StandardScaler()  # Standardize the features
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=5)

# Train the model through KNN Dataset
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train, y_train)

# Make predictions  
y_pred_knn = knn.predict(X_test)

# Calculate the accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn) 

# Output the result
print("KNN Accuracy score of the test data: ", knn_accuracy) 

# Print classification report 
print(classification_report(y_test, y_pred_knn)) 

# Input data for prediction the model
input_data = (3, 170, 69, 38, 13, 34, 0.47, 55)

# Create a DataFrame for the input data using the same columns as X
input_df = pd.DataFrame([input_data], columns=X.columns) 

# Standardize the input data directly from the DataFrame
std_data = scaler.transform(input_df)  
print("Standardized Input Data:", std_data)

# Make a prediction for the input data
prediction = knn.predict(std_data)   
print("Prediction According to the dataset: ", prediction)

# Output the result
if prediction[0] == 0:
    print('The Person is Not Diabetic')

else:
    print('The Person is Diabetic')
