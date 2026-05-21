# Credit Card Fraud Detection

This project builds and serves a machine learning-based credit card fraud detection app.
It includes a training script that prepares the data, handles class imbalance, trains multiple models, and saves the final artifacts, plus a Streamlit app for interactive fraud prediction and bulk CSV scoring.

## What I Built

I completed the project in two parts:

1. A model training pipeline in `main.py`.
2. A Streamlit inference app in `app.py`.

The training pipeline:

- Loads the transaction dataset.
- Splits features from the `Class` label.
- Standardizes the input features with `StandardScaler`.
- Splits the data into train and test sets.
- Trains a baseline logistic regression model.
- Evaluates the model with accuracy, precision, recall, F1, confusion matrix, and ROC-AUC.
- Uses SMOTE to address the heavy class imbalance in the fraud dataset.
- Retrains and compares three models on the resampled data:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Saves the trained models and scaler as `.pkl` files with `joblib`.

## Results

The final comparison across the resampled training set showed:

- Random Forest - F1: 0.855
- XGBoost - F1: 0.634
- Logistic Regression - F1: 0.099

These results were obtained after addressing class imbalance with SMOTE across the 30 PCA-transformed transaction features.

The Streamlit app:

- Loads the saved models and scaler.
- Lets the user choose between Logistic Regression, Random Forest, and XGBoost.
- Shows the fraud/non-fraud distribution from the dataset.
- Accepts manual transaction input for binary prediction.
- Displays feature importance for tree models and coefficient magnitude for logistic regression.
- Supports bulk fraud detection from an uploaded CSV file.
- Allows downloading the prediction results as a CSV.
- Includes a note explaining that the `V1` to `V28` columns are anonymized PCA components.

## Project Structure

- `main.py` - training and model comparison script.
- `app.py` - Streamlit user interface for prediction.
- `requirements.txt` - Python dependencies.
- `creditcard_sample.csv` - sample transaction data included in the repo.
- `lr_model.pkl` - saved logistic regression model.
- `rf_model.pkl` - saved random forest model.
- `xgb_fraud_model.pkl` - saved XGBoost model.
- `scaler.pkl` - saved feature scaler.

## How The Model Works

The dataset uses the following fields:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class`

`Class` is the target variable, where `1` represents fraud and `0` represents a legitimate transaction.

The feature columns are scaled before modeling because the transaction attributes are numeric and span different ranges. SMOTE is then applied to the training set so the models can learn from a more balanced fraud/non-fraud distribution.

## Models Trained

### Logistic Regression

I first trained a baseline logistic regression model on the original split data to establish a reference performance.

### SMOTE + Logistic Regression

I then applied SMOTE to the training set and retrained logistic regression with a higher iteration limit.

### Random Forest

A random forest model was trained on the SMOTE-balanced data using 200 trees.

### XGBoost

An XGBoost classifier was also trained on the SMOTE-balanced data with tuned learning and depth parameters.

## Streamlit App Features

The app in `app.py` provides:

- Model selection from the sidebar.
- A fraud distribution chart.
- Manual transaction prediction using `Time`, `Amount`, and generated placeholder values for `V1` to `V28`.
- Model-specific feature importance charts.
- CSV upload for batch prediction.
- CSV download of scored results.

## Setup

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run The Project

### Train The Models

```bash
python main.py
```

This script trains the models and writes the artifact files:

- `lr_model.pkl`
- `rf_model.pkl`
- `xgb_fraud_model.pkl`
- `scaler.pkl`

### Launch The Streamlit App

```bash
streamlit run app.py
```

## Data Notes

The code currently reads from `creditcard.csv`, while the repository contains `creditcard_sample.csv`.
If you want the scripts to run as-is, make sure the expected dataset filename exists or update the file path in both scripts.

## Important Implementation Details

- The scaler is fit before the train/test split in `main.py`, so the current pipeline is not fully leakage-free.
- The Streamlit app generates random values for `V1` to `V28` for manual single-transaction input, which is useful for testing the UI but not a realistic transaction entry flow.
- The bulk CSV upload expects a file with the same feature columns used by the scaler and models.

## Summary

This project demonstrates a complete end-to-end fraud detection workflow:

- data preprocessing
- imbalance handling with SMOTE
- model training and comparison
- interactive Streamlit fraud classification
- artifact export
- interactive fraud scoring in Streamlit

It is a solid baseline for a credit card fraud detection proof of concept and can be extended with better feature handling, proper dataset wiring, and production-grade model evaluation.
