# Bank Customer Churn Prediction App
### Project Overview
This project addresses the critical business problem of customer churn in the banking sector. By analyzing customer demographics and financial behavior, I built a machine learning application that predicts the likelihood of a customer leaving the bank (churning).

The final solution is a Streamlit Web App that allows bank managers to input customer data and receive a real-time churn risk assessment.

### Key Features
* **Predictive Modeling:** Uses a high-performance XGBoost Classifier.

* **Data Balancing:** Implemented SMOTE (Synthetic Minority Over-sampling Technique) to handle the significant class imbalance in the churn dataset.

* **Interactive UI:** A custom-built Streamlit dashboard for real-time "What-If" analysis.

* **Explainable AI:** Visualizes Feature Importance to show exactly what drives churn (e.g., Age, Number of Products, and Geography).

### Data Insights
From the Exploratory Data Analysis (EDA), several key drivers were identified (you can check the Predicting_Churn_Notebook for visualiations):

* **Geography (The "Germany" Spike):** Although France has the largest customer base, Germany shows a disproportionately high churn rate. The number of customers who exited in Germany is almost equal to those in France, despite having less than half the total population.

* **Gender Disparity:** There is a clear trend showing that Female customers are more likely to churn than Male customers. While the total count of male customers is higher, the absolute number of exits is greater among females.

* **Age Dynamics:** Risk increases significantly with age. While customers in their 30s have the highest volume of records, the ratio of "Exited" vs. "Stayed" begins to flip as customers enter their late 40s and 50s, marking them as high-risk demographics.

* **Product Usage & Stickiness:** Risk follows a non-linear trend where having exactly two products serves as the ultimate retention sweet spot for the bank. While customers with only one product account for the highest volume of churn records, the ratio of "Exited" vs. "Stayed" drastically flips as customers move to three or four products. In these multi-product categories, customers are almost certain to churn, marking them as the highest-risk demographic for the model.

### Tech Stack
* **Language:** Python

* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-learn (SMOTE).

* **Visualization:** Plotly, Seaborn, Matplotlib.

* **Deployment:** Streamlit.

### Installation & Setup
* **Clone the repository:**
Bash
git clone https://github.com/DivineOla-star/Bank-Churn-Prediction.git
cd Bank-Churn-Prediction

* **Create and Activate Virtual Environment:**
Bash
python -m venv venv
(*Windows:*
.\venv\Scripts\activate
OR *Mac/Linux:*
source venv/bin/activate)

* **Install Dependencies:**
Bash
pip install -r requirements.txt

* **Run the App:**
Bash
streamlit run app.py

### Project Structure
* **app.py:** The core Streamlit application script containing the UI and prediction logic.

* **Predicting_Churn_Notebook.ipynb:** The comprehensive google collab Notebook covering EDA, data preprocessing (SMOTE), and model training.

* **best_model.pkl:** The serialized XGBoost classifier, optimized via RandomizedSearchCV.

* **scaler.pkl:** The saved MinMaxScaler used to normalize numerical inputs like Balance and Salary.

* **feature_importance.xlsx:** An exported summary of feature scores used to generate the "Key Drivers" chart in the app.

* **xgboost_tree.png:** A pre-rendered visualization of the model's decision-making flow.

* **Churn_Modelling.csv:** The source dataset containing 10,000 customer records.
