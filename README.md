# Customer Churn Prediction – Decision Trees & Random Forests

Predicting customer churn for a streaming platform (StreamFlex) using Decision Tree 
and Random Forest classifiers, with full EDA, model tuning, and business recommendations.

## Overview

This project applies supervised machine learning to a customer churn dataset to identify 
at-risk subscribers and uncover the key factors driving cancellations. The analysis 
covers the full pipeline from data exploration to actionable business insights.

## Tasks

### Task 1 – Data Preparation & EDA
- Loaded and explored the `customer_churn.csv` dataset
- Verified no missing values or duplicate CustomerIDs
- Performed EDA including histograms, box plots, and a correlation heatmap
- Key finding: churn exceeds 78% once complaints surpass 3, and 95% when they exceed 7

### Task 2 – Decision Tree Classifier
- Separated features and target variable (`Churn`), applied one-hot encoding
- Split data 80/20 with stratification to preserve class balance
- Trained a baseline Decision Tree and evaluated with accuracy, precision, recall, F1, 
  and confusion matrix
- Tuned hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`) 
  using GridSearchCV with 5-fold cross-validation

### Task 3 – Random Forest Classifier
- Trained a Random Forest and compared performance against the tuned Decision Tree
- Random Forest outperformed on accuracy, recall, and F1; Decision Tree had higher precision
- Analyzed feature importances using Gini impurity reduction
- Top features: Watch_Time_Hours (0.137), Resolution_Time_Days (0.119), 
  Number_of_Complaints (0.118), Subscription_Length_Months (0.109)

### Task 4 – Business Insights & Recommendations
Three data-driven recommendations for StreamFlex:
1. **Boost engagement** – Flag low watch time and login frequency as early churn signals; 
   trigger personalized re-engagement campaigns
2. **Overhaul support** – Set resolution time targets under 13.5 days (Decision Tree 
   threshold); escalate accounts after 2 complaints before churn risk peaks
3. **Strengthen early loyalty** – Structured onboarding for new subscribers, discounted 
   premium upgrades, and proactive billing issue resolution

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (DecisionTreeClassifier, RandomForestClassifier, GridSearchCV)
- Matplotlib, Seaborn

## Results Summary

| Model | Accuracy | Recall | F1 |
|---|---|---|---|
| Baseline Decision Tree | — | — | — |
| Tuned Decision Tree | higher | higher | higher |
| Random Forest | best | best | best |

> Fill in your actual metric values from the notebook output.

## Dataset

`customer_churn.csv` – Contains customer demographics, usage behavior, support history, 
and subscription details for a fictional streaming platform.

## Team

- Gabriel Asencios
- Shashwat Shah
- Said El-Sherbiny
- Arvind Lakshmanan

Submitted for SOEN 471 – Big Data Analytics, Concordia University, March 2026.
