🏦 Bank Customer Churn Prediction App
📖 Project Overview
This project addresses the critical business problem of customer churn in the banking sector. By analyzing customer demographics and financial behavior, I built a machine learning application that predicts the likelihood of a customer leaving the bank (churning).

The final solution is a Streamlit Web App that allows bank managers to input customer data and receive a real-time churn risk assessment.

🚀 Key Features
Predictive Modeling: Uses a high-performance XGBoost Classifier.

Data Balancing: Implemented SMOTE (Synthetic Minority Over-sampling Technique) to handle the significant class imbalance in the churn dataset.

Interactive UI: A custom-built Streamlit dashboard for real-time "What-If" analysis.

Explainable AI: Visualizes Feature Importance to show exactly what drives churn (e.g., Age, Number of Products, and Geography).

📊 Data Insights
From the Exploratory Data Analysis (EDA), several key drivers were identified:

Geography: Customers in Germany show a significantly higher churn rate compared to France and Spain.

Demographics: Churn is notably higher among Female customers and older age groups.

Product Usage: Customers with only one product or more than two products (3 or 4) are more likely to churn.

🛠️ Tech Stack
Language: Python

Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE).

Visualization: Plotly, Seaborn, Matplotlib.

Deployment: Streamlit.

⚙️ Installation & Setup
Clone the repository:

Bash
git clone https://github.com/DivineOla-star/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction
Create and Activate Virtual Environment:

Bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
Install Dependencies:

Bash
pip install -r requirements.txt
Run the App:

Bash
streamlit run app.py
📁 Project Structure
app.py: The core Streamlit application script containing the UI and prediction logic.

Predicting_Churn_Notebook.ipynb: The comprehensive google collab Notebook covering EDA, data preprocessing (SMOTE), and model training.

best_model.pkl: The serialized XGBoost classifier, optimized via RandomizedSearchCV.

scaler.pkl: The saved MinMaxScaler used to normalize numerical inputs like Balance and Salary.

feature_importance.xlsx: An exported summary of feature scores used to generate the "Key Drivers" chart in the app.

xgboost_tree.png: A pre-rendered visualization of the model's decision-making flow.

Churn_Modelling.csv: The source dataset containing 10,000 customer records.