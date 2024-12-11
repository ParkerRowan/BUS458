import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

import pickle
import requests
from io import BytesIO

# URL to your model file on GitHub (replace with actual raw URL)
model_url = 'https://github.com/ParkerRowan/BUS458/blob/main/linear_regression_model2.pkl'

def load_model_from_github(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Ensure no HTTP errors (404, etc.)
        model = pickle.load(BytesIO(response.content))  # Use pickle to load model
        return model
    except requests.exceptions.RequestException as e:
        print(f"Error fetching model: {e}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model
model = load_model_from_github(model_url)

if model is None:
    st.error("Model could not be loaded. Please check the URL or try again later.")
else:
    st.success("Model loaded successfully!")

# Function to predict salary
def predict_salary(age, education_level, years_using_ml, years_experience, 
                   uses_database_lang, uses_statistical_lang, title, ml_expense):
    input_data = pd.DataFrame({
        'Age': [age],
        'EducationLevel': [education_level],
        'Years_Using_ML': [years_using_ml],
        'Years_Experience': [years_experience],
        'Uses_DatabaseLang': [uses_database_lang],
        'Uses_StatisticalLang': [uses_statistical_lang],
        'Title': [title],
        'ML_Expense': [ml_expense]
    })
    
    # Predict salary using the model
    prediction = model.predict(input_data)
    return prediction[0]

# Set up the Streamlit UI
st.title('Salary Prediction Model')
st.write("Please enter the following details to predict the salary:")

# Input fields for the user
age = st.number_input('Age', min_value=18, max_value=100, value=25)
education_level = st.number_input('Education Level', min_value=0, max_value=5, value=2)
years_using_ml = st.number_input('Years Using ML', min_value=0, max_value=20, value=3)
years_experience = st.number_input('Years of Experience', min_value=0, max_value=20, value=3)
uses_database_lang = st.selectbox('Uses Database Languages', options=[0, 1], index=0)
uses_statistical_lang = st.selectbox('Uses Statistical Languages', options=[0, 1], index=0)
title = st.number_input('Title', min_value=1, max_value=14, value=1)
ml_expense = st.number_input('ML Expense', min_value=0, max_value=10, value=2)

# Prediction button
if st.button('Predict Salary'):
    salary = predict_salary(age, education_level, years_using_ml, years_experience, 
                            uses_database_lang, uses_statistical_lang, title, ml_expense)
    st.write(f"The predicted salary is: ${salary:,.2f}")
