import streamlit as st
import pandas as pd
import pickle
import requests
from io import BytesIO

# URL to your raw model file on GitHub
model_url = 'https://github.com/ParkerRowan/BUS458/raw/main/trained_model.pkl'

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

# Load the model from GitHub
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

    # Predict salary using the trained model
    prediction = model.predict(input_data)  # This should be the trained model, not a numpy array
    return prediction[0]

# Set up the Streamlit UI
st.set_page_config(page_title="Salary Prediction Model", page_icon="💼", layout="wide")

# Add a title and description
st.title("💼 Salary Prediction Model")
st.markdown("""
    Please enter the following details to predict the salary of an individual 
    based on their experience, education, and other factors.
""")

# Input fields for the user (styled with markdown and emojis)
st.subheader("Input Information")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=18, max_value=100, value=25, help="Age of the individual.")
    education_level = st.selectbox('Education Level', options=list(range(0, 6)), index=2, help="Education level on a scale of 0 to 5.")
    years_using_ml = st.selectbox('Years Using ML', options=list(range(0, 9)), index=3, help="Number of years using machine learning.")
    uses_database_lang = st.selectbox('Uses Database Languages', options=[0, 1], index=0, help="Whether the individual uses database languages.")
    title = st.selectbox('Title', options=list(range(0, 15)), index=0, help="The job title of the individual.")
    
with col2:
    years_experience = st.selectbox('Years of Experience', options=list(range(0, 7)), index=3, help="Years of professional experience.")
    uses_statistical_lang = st.selectbox('Uses Statistical Languages', options=[0, 1], index=0, help="Whether the individual uses statistical languages.")
    ml_expense = st.selectbox('ML Expense', options=list(range(0, 6)), index=2, help="The ML expense on a scale from 0 to 5.")
    
# Add space and a button to trigger prediction
st.markdown("---")
if st.button('Predict Salary'):
    salary = predict_salary(age, education_level, years_using_ml, years_experience, 
                            uses_database_lang, uses_statistical_lang, title, ml_expense)
    if salary is not None:
        st.markdown(f"### The predicted salary is: **${salary:,.2f}**")
    else:
        st.write("There was an issue with the prediction. Please check the input data.")
