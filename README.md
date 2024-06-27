# CardGuardML - Credit Card Fraud Detection Using Machine Learning


## Overview

In this project, we aim to predict fraudulent credit card transactions using advanced machine learning techniques. Given the critical nature of this task, our goal is to develop a robust system that can accurately identify fraudulent transactions, thereby helping financial institutions minimize financial losses and enhance customer trust.

## Problem Statement

Credit card fraud is a significant issue that poses a threat to both banks and customers. Our objective is to build a sophisticated fraud detection system using machine learning algorithms to identify and prevent fraudulent transactions. We leverage a highly imbalanced dataset from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains 284,807 transactions, of which only 492 are fraudulent.

## Business Context

For financial institutions, detecting and preventing fraud is crucial to maintaining profitability and customer trust. According to the [Nilson Report](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_Issue_1164.pdf), banking fraud is projected to reach $30 billion globally by 2020. Our machine learning approach aims to provide a proactive fraud detection mechanism, reducing manual reviews, chargebacks, and fraudulent transactions.

## Dataset Description

The dataset used for this project includes credit card transactions made by European cardholders over two days in September 2013. It comprises 284,807 transactions with 30 feature columns:

- **Time**: Seconds elapsed between the first transaction and each subsequent transaction.
- **V1 to V28**: Principal components obtained via PCA to protect confidentiality.
- **Amount**: Transaction amount.
- **Class**: Target variable (0 for non-fraud, 1 for fraud).

## Project Pipeline

The project follows these key steps:

1. **Data Understanding**: Load and explore the dataset to understand its structure and features.
2. **Exploratory Data Analysis (EDA)**: Perform univariate and bivariate analyses to uncover patterns and insights.
3. **Data Preprocessing**: Handle missing values, outliers, and data imbalance. Normalize data if necessary.
4. **Train/Test Split**: Split the data into training and testing sets. Use k-fold cross-validation for model evaluation.
5. **Model Building**: Experiment with various machine learning models (Logistic Regression, Random Forest, XGBoost, etc.) and deep learning models (Neural Networks).
6. **Hyperparameter Tuning**: Optimize model performance through hyperparameter tuning.
7. **Model Evaluation**: Evaluate models using appropriate metrics (Precision, Recall, ROC-AUC) considering the imbalanced nature of the data.
8. **Results and Insights**: Summarize findings and visualize key metrics and performance.

## Techniques and Tools

- **Data Handling**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning Models**: Scikit-learn, XGBoost
- **Deep Learning Models**: TensorFlow, Keras
- **Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)

## Key Findings

- **Data Imbalance**: The dataset is highly imbalanced with only 0.172% fraudulent transactions.
- **Model Performance**: Various models were tested, and the best results were achieved using the SMOTE technique combined with a Neural Network model.
- **Evaluation Metrics**: Precision, Recall, and ROC-AUC were the primary metrics used to evaluate model performance due to the importance of accurately detecting fraudulent transactions.

## Model Performance

The best performing model was a Neural Network trained with SMOTE-oversampled data, achieving the following metrics:

- **Precision**: High precision ensures fewer false positives, reducing unnecessary manual reviews.
- **Recall**: High recall ensures most fraudulent transactions are detected, minimizing financial losses.
- **ROC-AUC**: The ROC-AUC score provides a balanced measure of performance, considering both true positive and false positive rates.

## Conclusion

Our project successfully developed a machine learning-based credit card fraud detection system that can significantly improve the efficiency and effectiveness of fraud detection processes in financial institutions. The use of SMOTE for handling data imbalance and the implementation of advanced machine learning and deep learning models were key to achieving high performance.

## Future Work

- **Feature Engineering**: Explore additional features and domain-specific knowledge to enhance model performance.
- **Real-time Detection**: Implement the model for real-time fraud detection.
- **Model Explainability**: Improve model interpretability to provide actionable insights to business stakeholders.

## References

- [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- [Nilson Report](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_Issue_1164.pdf)
- [SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

