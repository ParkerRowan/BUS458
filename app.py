import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

# URL to your raw model file on GitHub
model_url = 'https://github.com/ParkerRowan/BUS458/blob/main/trained_model.pkl'

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

# Load the trained model from GitHub
model = load_model_from_github(model_url)

if model is None:
    st.error("Model could not be loaded. Please check the URL or try again later.")
else:
    st.success("Model loaded successfully!")

# Function to predict salary
def predict_salary(age, education_level, years_using_ml, years_experience, 
                   uses_database_lang, uses_statistical_lang, title, ml_expense):
    # Prepare the input data as a DataFrame
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

    # Convert input_data into a 2D numpy array
    input_data = input_data.values  # This converts the DataFrame to a 2D numpy array

    # Ensure the model is the correct trained model
    if not hasattr(model, 'predict'):
        st.error("Model is not loaded properly. Please check the model.")
        return None

    # Predict salary using the trained model
    prediction = model.predict(input_data)  # This should be the trained model, not a numpy array
    return prediction[0]

# Set up the Streamlit UI
st.title('Salary Prediction Model')
st.write("Please enter the following details to predict the salary:")

# Input fields for the user
# Age: Number input with + and - buttons
age = st.number_input('Age', min_value=18, max_value=100, value=25)

# Other fields: Dropdown menus for categorical selections
education_level = st.selectbox('Education Level', options=list(range(0, 6)), index=2)  # Range 0-5
years_using_ml = st.selectbox('Years Using ML', options=list(range(0, 9)), index=3)  # Range 0-8
years_experience = st.selectbox('Years of Experience', options=list(range(0, 7)), index=3)  # Range 0-6
uses_database_lang = st.selectbox('Uses Database Languages', options=[0, 1], index=0)
uses_statistical_lang = st.selectbox('Uses Statistical Languages', options=[0, 1], index=0)
title = st.selectbox('Title', options=list(range(0, 15)), index=0)  # Range 0-14
ml_expense = st.selectbox('ML Expense', options=list(range(0, 6)), index=2)  # Range 0-5

# Prediction button
if st.button('Predict Salary'):
    salary = predict_salary(age, education_level, years_using_ml, years_experience, 
                            uses_database_lang, uses_statistical_lang, title, ml_expense)
    if salary is not None:
        st.write(f"The predicted salary is: ${salary:,.2f}")
    else:
        st.write("There was an issue with the prediction. Please check the input data.")
