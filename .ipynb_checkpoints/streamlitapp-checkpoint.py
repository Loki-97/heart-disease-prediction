import streamlit as st
import joblib
import numpy as np

# ✅ Set page config FIRST
st.set_page_config(page_title="Heart Disease Prediction", page_icon="💓", layout="wide")

# ✅ CSS for Background (Choose between Gradient or Image)
st.markdown(
    """
    <style>
    .stApp {
        background: url("https://res.cloudinary.com/dh8oiorxd/image/upload/v1740080575/freepik__the-style-is-candid-image-photography-with-natural__88923_ndrovq.png") no-repeat center center fixed;
        background-size: cover;
    }
    
    /* Text Styling for Better Visibility */
    h1, h2, h3, p, label {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
    }

     div[data-testid="stTooltipContent"] {
    background-color: black !important;  /* Dark background */
    color: white !important;  /* White text */
    font-size: 14px;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 2px 2px 6px rgba(0, 0, 0, 0.5);
}   

 button[kind="tooltip"] svg {
        color: black !important;  /* Change icon color to black */
        background: white !important;  /* Add a white background */
        border-radius: 50% !important;  /* Make it circular */
        padding: 5px !important;  /* Add spacing */
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.3) !important;  /* Add a shadow */
    }

    /* If the icon is inside a div */
    div[data-testid="stTooltipIconContainer"] svg {
        color: black !important;
        background: white !important;
        border-radius: 50% !important;
        padding: 5px !important;
        box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.3) !important;
    }
    

    /* Button Styling */
    div.stButton > button {
        background-color: #28a745 !important; /* Green color */
        color: white !important;
        border-radius: 10px;
        width: 200px;
        font-size: 18px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #218838 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Load FontAwesome icons
st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">',
    unsafe_allow_html=True
)

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

# Set up UI with FontAwesome icons
st.markdown("<h1 style='text-align: center;'><i class='fas fa-heartbeat'></i> Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.write("Enter patient details below to predict heart disease risk.")

# Layout with icons
col1, col2 = st.columns(2)

with col1:
    st.markdown("⚠️ **Chest Pain Type** <i class='fas fa-heart'></i>", unsafe_allow_html=True)
    cp = st.selectbox(
    "⚠️ **Chest Pain Type**",  
    ["0 - Typical Angina", "1 - Atypical Angina", "2 - Non-Anginal Pain", "3 - Asymptomatic"],  
    help="Type of chest pain."
    )

    # Convert to integer for model input
    cp = int(cp.split(" - ")[0])




    st.markdown("💓 **Max Heart Rate** <i class='fas fa-heartbeat'></i>", unsafe_allow_html=True)
    thalach = st.slider("", 60, 200, 120, help="Maximum heart rate.")

with col2:
    st.markdown("📈 **Slope of ST Segment** <i class='fas fa-chart-line'></i>", unsafe_allow_html=True)
    slope = st.selectbox("📈 **Slope of ST Segment**",  
                     ["0 - Upsloping", "1 - Flat", "2 - Downsloping"],  
                     help="Slope of the ST segment in ECG.")

    slope = int(slope.split(" - ")[0])  # Extract number


    st.markdown("📉 **ST Depression** <i class='fas fa-wave-square'></i>", unsafe_allow_html=True)
    oldpeak = st.number_input("", help="ST depression level.")

    st.markdown("🏃 **Exercise Induced Angina** <i class='fas fa-walking'></i>", unsafe_allow_html=True)
    exang = st.selectbox("🏃 **Exercise Induced Angina**",  
                     ["0 - No", "1 - Yes"],  
                     help="Angina triggered by exercise.")

    exang = int(exang.split(" - ")[0])  # Extract number

st.markdown("---")  # Adds a separator

# Predict button with an icon
if st.button("🔮 Predict"):
    input_data = np.array([[cp, thalach, slope, oldpeak, exang]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error(" High risk of Heart Disease! ", icon="⚠️")
    else:
        st.success(" Low risk of Heart Disease. ", icon="✅")