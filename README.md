📊 Customer Churn Prediction Dashboard

An end-to-end machine learning project that predicts customer churn using telecom customer data and presents insights through an interactive Streamlit dashboard.

🔍 Project Overview

Customer churn is a major business problem where customers stop using a company’s services.
This project analyzes customer behavior, identifies key churn drivers, and predicts whether a customer is likely to churn using machine learning.

The final output is an interactive dashboard that allows users to:

Explore churn trends visually

View key churn metrics

Predict churn for individual customers

🧠 Key Features

Data cleaning and preprocessing (handling missing values, encoding categorical data)

Exploratory Data Analysis (EDA) to understand churn patterns

Machine learning models for churn prediction

Model evaluation using Accuracy, Precision, Recall, F1 Score, and ROC-AUC

Interactive Streamlit dashboard with real-time prediction

Feature importance visualization for explainability

🛠️ Tech Stack / Tools

Programming Language: Python

Libraries: Pandas, NumPy, Scikit-learn

Visualization: Plotly, Matplotlib, Seaborn

Dashboard: Streamlit

Model Persistence: Joblib

📈 Machine Learning Models

Logistic Regression (baseline and final deployed model)

Random Forest (tested and compared)

Best ROC-AUC achieved: ~0.83

▶️ How to Run the Project

1️⃣ Clone the repository
git clone https://github.com/H-arshini-g/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

2️⃣ Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the dashboard
streamlit run app/dashboard.py


The app will open automatically in your browser at:

http://localhost:8501

📊 Insights Gained

Customers with month-to-month contracts have the highest churn rate

High monthly charges significantly increase churn probability

New customers (low tenure) are more likely to churn

Support-related services (TechSupport, OnlineSecurity) help reduce churn
