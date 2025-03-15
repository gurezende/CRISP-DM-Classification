# Streamlit App for Financial App
# Predicting the probability of a customer to convert when offered a financial product (direct term deposit) via a phone call.

# Imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from my_funcs import *


# ---------- Load the model ----------
try:
    filename = 'Scripts/model6.pkl'
    model = pickle.load(open(filename, 'rb')) 
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'model.pkl' is in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# -------------- Define Predictor Inputs in Streamlit UI --------------
st.set_page_config(layout="centered")
st.title("Term Deposit Conversion Probability Predictor")
st.write("Enter customer information to predict the probability of subscribing to a term deposit.")

# Create two columns
col1, col2 = st.columns(2)

# Input fields
with col1:
    default = st.selectbox("Has credit in default?", ['No', 'Yes'])
    housing = st.selectbox("Has housing loan?", ['Yes', 'No'])
    loan = st.selectbox("Has personal loan?", ['Yes', 'No'])
    contact = st.selectbox("Contact communication type", ['cellular', 'telephone'])
    month = st.selectbox("Month of last contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    
with col2:
    campaign = st.slider("Number of contacts performed during this campaign", 0, 50, 1)
    day = st.slider("Day of the month", 1, 31, 15)
    pdays = st.slider("Number of days that passed by after the client was last contacted from a previous campaign", 0, 100, 0)

#-------- Preprocess Input Data ----------
# Create a dictionary to hold the input values
input_data = pd.DataFrame({
    'default': default,
    'housing': housing,
    'loan': loan,
    'day': day,
    'contact_cellular': 1 if contact == 'cellular' else 0,
    'contact_telephone': 1 if contact == 'telephone' else 0,
    'month': month,
    'campaign': campaign,
    'pdays': pdays,
    'y':99
}, index=[0])

# Create a DataFrame from the input data
input_df = prepare_data_simpler_streamlit(input_data)


# ----------------- Make Prediction and Display Results ----------------
if st.button("Predict"):
    try:
        prediction_proba = model.predict_proba(input_df)[0][1]  # Probability of class 1 (conversion)
        prediction_percentage = round(prediction_proba * 100, 2)

        st.success(f"The predicted probability of the customer subscribing to a term deposit is: **{prediction_percentage}%**",
                   icon="💻")

        # Visualization
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(y=['Probability'], x=[prediction_percentage], ax=ax, color='forestgreen')
        plt.axvline(50, color='gray', linestyle='--', linewidth=2)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Conversion Probability")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

