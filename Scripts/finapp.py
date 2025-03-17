# Streamlit App for Financial App
# Predicting the probability of a customer to convert when offered a financial product (direct term deposit) via a phone call.
#%%
# Imports
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from my_funcs import *

# Set page config
st.set_page_config(layout="centered")

# ---------- Function to cache the model ----------

@st.cache_resource
def load_model_from_mlflow():
    """Loads the model from MLflow."""

    # Making connection to the MLFlow server
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Start a MLFlow Client to get the latest model version
    client = mlflow.client.MlflowClient()

    # Getting latest version
    version = client.get_registered_model("Catboost_Simpler_Cols").latest_versions[0].version

    # Import the latest model version
    try:
        model = mlflow.catboost.load_model(f"models:/Catboost_Simpler_Cols/{version}")
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    return model


# ---------- Load the model ----------

model = load_model_from_mlflow()

#-------------------------------------------------------

# -------------- Define Predictor Inputs in Streamlit UI --------------
st.title("Term Deposit Conversion Probability Predictor")
st.write("Enter customer information to predict the probability of subscribing to a term deposit.")

# Create two columns
col1, col2, col3, col4 = st.columns(4)

# Input fields
with col1:
    default = st.pills("Has credit in default?", ['Yes', 'No'], default='No', key='default')
    
with col2:
    loan = st.pills("Has personal loan?", ['Yes', 'No'], key='loan')

with col3:
    housing = st.pills("Has housing loan?", ['Yes', 'No'], key='housing')

with col4:
    contact = st.pills("Contact Type", ['cellular', 'telephone'], key='contact')

month = st.pills("Month of last contact", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
day = st.slider("Day of the month", 1, 31, 15)
campaign = st.slider("Contacts during this campaign", 0, 50, 1)
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


        # ---------- Visualization ----------

        # Data for parties and their probs counts
        parties = ['Not Convert', ' Probability of Convertion']
        probs = [(100-int(prediction_percentage))*10, int(prediction_percentage)*10]

        # Assign colors to parties
        colors = ['#d3d3d3', '#57b45f']
        
        # Define levels (radius values)
        num_levels = 15
        levels = generate_levels(num_levels, min_radius=0.0, max_radius=1)

        # Generate points per level
        total_probs = sum(probs)
        points_per_level = generate_points(levels, total_probs)

        theta_start, theta_end = 0, 180
        # Generate radii and theta values
        radii_sorted, theta_sorted = generate_radii_theta(levels, points_per_level, theta_start, theta_end)

        # Create the parliament chart
        fig = create_parliament_chart(parties, probs, colors, radii_sorted, theta_sorted, marker_size=6)

        # Set up the layout for the chart
        title = "Conversion Probability"
        if prediction_percentage > 50:
            subtitle = f"This customer has a good chance of subscribing to a term deposit.Predicted probability: {prediction_percentage}%"
        else:
            subtitle = f"The customer has a low chance of subscribing to a term deposit.Predicted probability: {prediction_percentage}%"

        setup_layout(fig, title, subtitle)

        # Show the figure
        st.plotly_chart(fig, theme=None)



        # # Visualization
        # fig, ax = plt.subplots(figsize=(6, 4))
        # sns.barplot(y=['Probability'], x=[prediction_percentage], ax=ax, color='forestgreen')
        # plt.axvline(50, color='gray', linestyle='--', linewidth=2)
        # ax.set_xlim(0, 100)
        # ax.set_xlabel("Probability (%)")
        # ax.set_title("Conversion Probability")
        # st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


# %%
