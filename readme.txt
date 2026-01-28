SYSTEM DESIGN

🟦 Phase 1 – Sprint 1 (1 Hour): System Design
Objective

To clearly understand the business problem, frame it as a machine learning classification task, select and justify the dataset, define the target label, and design a high-level end-to-end solution architecture.

1.1 Restate the Problem in Our Own Words

In e-commerce platforms, a significant number of product orders are returned after delivery. Product returns lead to increased reverse logistics costs, inventory handling issues, and reduced profitability.

The goal of this project is to predict whether an order will be returned or not at the time of order placement, using only the information available before delivery. By identifying high-risk orders early, businesses can take preventive actions such as improving product descriptions, adjusting return policies, or flagging risky transactions.

1.2 Machine Learning Framing

Learning Type: Supervised Learning
Problem Type: Binary Classification
Prediction Objective:
Predict if an order will be returned or not
Prediction Timing: At order placement (pre-delivery)

Output:

1 → Returned
0 → Not Returned

1.3 Dataset Scouting and Choice
Dataset Selected
Synthetic Dataset for E-Commerce Return Analysis (Kaggle)

Dataset Context
Domain: Retail / E-commerce
Granularity: Order-level records
Nature: Synthetic but realistic business data
Justification for Selection

This dataset was chosen because it satisfies all hackathon requirements:

Belongs to a retail/e-commerce context
Contains explicit return-related information
Has sufficient rows for machine learning modeling
Includes relevant order-time features such as:
Product category
Price
Discount percentage
Payment method
Shipping type
Order and customer attributes
Suitable for classification and demo-based evaluation

1.4 Label Definition Strategy
Target Variable

The target variable is returned, defined as:
returned = 1 → Order was returned or cancelled
returned = 0 → Order was successfully completed and not returned

Label Source
The label is derived from one of the following dataset columns (depending on availability):
return_status
is_returned

Data Leakage Prevention

Only information available before delivery is used.
Post-delivery fields such as return reason, refund amount, refund date, or delivery status are excluded to avoid data leakage.

1.5 Feature Design (Initial)
Included Features (Order-Time Only)
Product-Level Features
Product category
Product price
Discount percentage
Order-Level Features
Payment method
Shipping type
Order date-derived features (day, month, weekday)
Customer-Level Features (If Available & Safe)
Customer segment
Historical order attributes

Excluded Features (Leakage Prevention)

Return reason
Refund amount
Refund date
Delivery status
Any post-delivery or post-return information

1.6 High-Level System Architecture

End-to-End Flow


High Level Architecture-

Raw Retail Dataset 
        ↓
Data Cleaning & EDA 
        ↓
Missing values, Feature Engineering, Encoding, Normalization
        ↓
Model Training 
        ↓
Logistic Regression(Baseline) -> Random Forest -> XGBoost 
        ↓
Model Evaluation, Accuracy, Precision, ROC-AUC, F1, Recall
        ↓ 
Save Best Model among all, used joblib
        ↓
UI / Endpoint (Streamlit App)
        ↓
User enters order details 
        ↓
Get return-risk score 
        ↓
Calibrated Predicted Probabilities
        ↓
Visualized Feature Importance
        ↓
Implemented different thresholds for classification


1.7 Technology Stack

Programming Language: Python
Data Processing: Pandas, NumPy
Visualization: Matplotlib
Machine Learning: Scikit-learn
Model Used: Logistic Regression

UI for Demo: Streamlit
Development Environment: IDE (VS Code )

Role Split : Pushpendra Yadav - Dataset Sourcing and System Design
             Sahil Anand -  EDA, Model Selection & Training
             Lakshya Shukla - Front End UI and Endpoint Integration