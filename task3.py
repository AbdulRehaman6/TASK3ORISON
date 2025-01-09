import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = 'Housing.csv'
housing_data = pd.read_csv(file_path)

# Separate features and target variable
x = housing_data.drop(columns=['price'])
y = housing_data['price']

# Identify categorical and numerical columns
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']
numerical_cols = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

# Apply Label Encoding for categorical data
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Streamlit App
st.title("House Price Prediction App")

# Collect user inputs for prediction
st.subheader("Input House Features")
input_data = {}

# Collect numerical inputs
for col in numerical_cols:
    input_data[col] = st.number_input(f"Enter {col}", min_value=0, step=1)

# Collect categorical inputs
for col in categorical_cols:
    input_data[col] = st.selectbox(f"Select {col}", label_encoders[col].classes_)

# Transform input data into DataFrame
input_df = pd.DataFrame([input_data])

# Ensure the transformation of categorical variables using label encoders
for col in categorical_cols:
    input_df[col] = label_encoders[col].transform(input_df[col])

# Ensure input_df has the same column order as the model's training data
input_df = input_df[model.feature_names_in_]

# Predict and display result if button is clicked
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Predicted House Price")
        st.write(f"₹ {prediction:,.2f}")
    except Exception as e:
        st.write(f"Error occurred: {str(e)}")
