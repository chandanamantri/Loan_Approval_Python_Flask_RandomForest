# Loan Approval Prediction — Python + Flask + Random Forest

## Technologies
- HTML5
- CSS3
- JavaScript
- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Random Forest Classifier

## Project flow
User → Web Form → Flask Backend → Preprocessing → Random Forest → Prediction → Result

## Run on Windows

1. Open Command Prompt in this project folder.
2. Create a virtual environment:
   `python -m venv venv`
3. Activate it:
   `venv\Scripts\activate`
4. Install packages:
   `pip install -r requirements.txt`
5. Train/create the model:
   `python train_model.py`
6. Start the web application:
   `python app.py`
7. Open:
   `http://127.0.0.1:5000`

## Important
A small built-in sample dataset is included inside `train_model.py` so the project runs immediately. If you have the original Loan Prediction CSV, place it at:

`data/loan_data.csv`

The CSV should contain the standard fields:
Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
Property_Area, Loan_Status

## Algorithm
Random Forest Classifier is used for loan approval prediction. It combines multiple decision trees and uses their combined predictions to classify an application as Approved or Rejected.
