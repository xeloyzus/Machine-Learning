
Explainable Machine Learning for Gallstone Disease

A machine learning project designed to predict the presence of gallstone disease using clinical tabular data and provide interpretable insights into the underlying features driving the predictions.

  Project Overview

Gallstone disease is a common medical condition closely linked to demographic, metabolic, body-composition, and biochemical characteristics. While standard machine learning models can achieve high classification accuracy, they often function as "black boxes."
This project addresses this limitation by focusing on Explainable AI (XAI)—not only predicting the disease but also uncovering why the model makes a particular prediction.

   Objectives

1. Classification: Develop a machine learning model to accurately distinguish between patients with and without gallstone disease.
2. Explainability: Investigate and interpret which clinical characteristics contribute most to the model's predictions.

  Dataset Details

The dataset used in this project is sourced from the UCI Machine Learning Repository:
🔗 Gallstone-1 Dataset (UCI)
• Data Type: Tabular
• Cohort & Timeframe: 319 outpatients from Ankara VM Medical Park (collected between June 2022 and June 2023).
• Class Balance: Fully balanced classes (161 patients diagnosed with gallstones, 158 without).
• Missing Values: None (Clean dataset).
• Ethics Approval: Ankara City Hospital (E2-23-4632).

Features (38 Total)

The dataset contains a rich set of clinical variables categorized into four main domains:
• Demographics: Age, sex, height, weight, BMI.
• Bioimpedance: Total water, extracellular water, intracellular water, muscle mass, fat mass, protein, visceral fat area.
• Hepatic Metrics: Hepatic fat measurements.
• Laboratory Tests: Glucose, lipids (cholesterol profile), AST, ALT, ALP, creatinine, GFR, CRP, hemoglobin, vitamin D.

 Reference & Literature

This project is built upon the methodology and findings outlined in the introductory paper:
• Paper Link: Early Prediction of Gallstone Disease with a Machine Learning Approach (Medicine Journal)

 Getting Started (Optional Setup Template)


Prerequisites

Ensure you have the following libraries installed:
bash
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn
Use code with caution.

Quick Start

1. Clone the repository:bash
git clone https://github.com
Use code with caution.
2. Run the notebook/script to train the model and generate explanations:bash
python main.py
Use code with caution.
