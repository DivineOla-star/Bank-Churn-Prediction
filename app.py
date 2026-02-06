import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

import pandas as pd
import plotly.express as px
import pickle


#""##6. Streamlit App"""

# Title and description
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("""
This app predicts whether a bank customer is likely to churn based on their profile information.
Adjust the parameters in the sidebar and click **Predict** to see the results.
""")

# Load the model and scaler
@st.cache_resource
def load_models():
    try:
        with open("best_model.pkl", "rb") as file:
            model = pickle.load(file)
        with open("scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()

model, scaler = load_models()

# Define feature names that match your trained model
# Based on your preprocessing, these should be the features:
feature_names = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'EstimatedSalary', 'Geography_France', 'Geography_Germany', 
    'Geography_Spain', 'Gender_Female', 'Gender_Male', 
    'HasCrCard_0', 'HasCrCard_1', 'IsActiveMember_0', 'IsActiveMember_1'
]

# Create sidebar for user inputs
st.sidebar.header("📊 Customer Information")

# Collect user inputs with more user-friendly labels
user_inputs = {}

# Numerical inputs
user_inputs['CreditScore'] = st.sidebar.slider("Credit Score", 300, 850, 600)
user_inputs['Age'] = st.sidebar.slider("Age", 18, 100, 30)
user_inputs['Tenure'] = st.sidebar.slider("Tenure (years with bank)", 0, 10, 2)
user_inputs['Balance'] = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 8000.0, 100.0)
user_inputs['NumOfProducts'] = st.sidebar.slider("Number of Products", 1, 4, 2)
user_inputs['EstimatedSalary'] = st.sidebar.number_input("Estimated Salary", 0.0, 300000.0, 60000.0, 1000.0)

# Categorical inputs
st.sidebar.subheader("Customer Details")

# Geography - using radio buttons for better UX
geo = st.sidebar.radio("Geography", ["France", "Germany", "Spain"])
user_inputs['Geography_France'] = 1 if geo == "France" else 0
user_inputs['Geography_Germany'] = 1 if geo == "Germany" else 0
user_inputs['Geography_Spain'] = 1 if geo == "Spain" else 0

# Gender
gender = st.sidebar.radio("Gender", ["Female", "Male"])
user_inputs['Gender_Female'] = 1 if gender == "Female" else 0
user_inputs['Gender_Male'] = 1 if gender == "Male" else 0

# Has Credit Card
has_card = st.sidebar.radio("Has Credit Card?", ["Yes", "No"])
user_inputs['HasCrCard_0'] = 1 if has_card == "No" else 0
user_inputs['HasCrCard_1'] = 1 if has_card == "Yes" else 0

# Is Active Member
active_member = st.sidebar.radio("Is Active Member?", ["Yes", "No"])
user_inputs['IsActiveMember_0'] = 1 if active_member == "No" else 0
user_inputs['IsActiveMember_1'] = 1 if active_member == "Yes" else 0

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_inputs])

# Ensure the columns are in the correct order
input_df = input_df[feature_names]

# Scale numerical features
scale_vars = ["CreditScore", "EstimatedSalary", "Tenure", "Balance", "Age", "NumOfProducts"]
input_df_scaled = input_df.copy()
input_df_scaled[scale_vars] = scaler.transform(input_df[scale_vars])



# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("📈 Prediction")
    
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(input_df_scaled)[0]
            prediction_proba = model.predict_proba(input_df_scaled)[0]
            
            # Display results
            if prediction == 1:
                st.error(f"🚨 High Risk of Churning!")
                st.metric("Churn Probability", f"{prediction_proba[1]*100:.1f}%")
            else:
                st.success(f"✅ Low Risk of Churning")
                st.metric("Retention Probability", f"{prediction_proba[0]*100:.1f}%")
            
            # Show probability breakdown
            st.subheader("Probability Breakdown")
            prob_df = pd.DataFrame({
                'Status': ['Retain', 'Churn'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            
            fig = px.bar(prob_df, x='Status', y='Probability', 
                        color='Status',
                        color_discrete_map={'Retain': 'green', 'Churn': 'red'},
                        text_auto='.1%')
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

with col2:
    st.header("⚖️ Feature Importance")
    
    try:
        # Load feature importance from Excel
        feature_importance_df = pd.read_excel("feature_importance.xlsx")
        
        # Plot feature importance
        fig = px.bar(feature_importance_df.sort_values('Feature Importance Score', ascending=True),
                     x='Feature Importance Score', 
                     y='Feature',
                     orientation='h',
                     title="Top Features Influencing Churn",
                     color='Feature Importance Score',
                     color_continuous_scale='RdYlGn_r')
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.info("Feature importance data not available. Please run the model training script first.")

# --- NEW SECTION: DECISION LOGIC VISUALIZATION FIX ---
st.markdown("---") # Visual separator
st.header("🌳 Decision Logic Visualization")
try:
    # This replaces the crashing graphviz logic with a stable image display
    st.image("xgboost_tree.png", 
             caption="XGBoost Decision Tree - Model Logic Flow", 
             use_container_width=True)
except FileNotFoundError:
    st.warning("Tree image not found. Please save 'xgboost_tree.png' to the project folder.")


# Display the input values for transparency
st.sidebar.header("📋 Input Summary")
st.sidebar.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Instructions:**
    1. Adjust customer parameters in the sidebar
    2. Click **Predict Churn** button
    3. View prediction results and feature importance
    """
)