import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

os.makedirs("model", exist_ok=True)

# Self-contained training data based on the standard Loan Prediction
# dataset feature structure. Replace data/loan_data.csv with your own
# dataset when available.
csv_path = "data/loan_data.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Small representative dataset so the project runs immediately.
    rows = [
        ["Male","Yes","0","Graduate","No",5849,0,128,360,1,"Urban","Y"],
        ["Male","Yes","1","Graduate","No",4583,1508,128,360,1,"Rural","N"],
        ["Male","Yes","0","Graduate","Yes",3000,0,66,360,1,"Urban","Y"],
        ["Female","No","0","Graduate","No",4100,0,100,360,1,"Semiurban","Y"],
        ["Male","Yes","2","Not Graduate","No",2500,1800,90,360,0,"Rural","N"],
        ["Male","No","0","Graduate","No",6000,0,150,360,1,"Urban","Y"],
        ["Female","Yes","1","Graduate","No",3500,1200,110,360,1,"Semiurban","Y"],
        ["Male","Yes","3+","Not Graduate","Yes",2200,900,75,360,0,"Rural","N"],
        ["Female","No","0","Graduate","No",5000,0,125,360,1,"Urban","Y"],
        ["Male","Yes","1","Graduate","No",2700,1000,95,360,0,"Rural","N"],
        ["Male","Yes","2","Graduate","No",7000,0,200,360,1,"Urban","Y"],
        ["Female","No","0","Not Graduate","No",1800,0,60,360,0,"Rural","N"],
        ["Male","No","0","Graduate","No",4200,500,120,360,1,"Semiurban","Y"],
        ["Female","Yes","2","Graduate","Yes",3200,1300,105,360,1,"Urban","Y"],
        ["Male","Yes","3+","Not Graduate","No",2600,0,80,360,0,"Rural","N"],
        ["Male","Yes","0","Graduate","No",4800,1000,130,360,1,"Semiurban","Y"],
        ["Female","No","1","Graduate","No",3900,0,115,360,1,"Urban","Y"],
        ["Male","Yes","1","Not Graduate","No",2300,1200,85,360,0,"Rural","N"],
        ["Male","No","0","Graduate","Yes",5500,0,140,360,1,"Urban","Y"],
        ["Female","Yes","0","Graduate","No",4500,800,125,360,1,"Semiurban","Y"],
        ["Male","Yes","2","Not Graduate","No",2800,700,100,360,0,"Rural","N"],
        ["Male","Yes","0","Graduate","No",6200,0,155,360,1,"Urban","Y"],
        ["Female","No","0","Graduate","No",3600,0,90,360,1,"Semiurban","Y"],
        ["Male","Yes","3+","Graduate","Yes",3100,1400,120,360,0,"Rural","N"],
    ]
    cols = ["Gender","Married","Dependents","Education","Self_Employed",
            "ApplicantIncome","CoapplicantIncome","LoanAmount",
            "Loan_Amount_Term","Credit_History","Property_Area","Loan_Status"]
    df = pd.DataFrame(rows, columns=cols)

X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

categorical = ["Gender","Married","Dependents","Education","Self_Employed","Property_Area"]
numeric = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ]), numeric)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=8,
        class_weight="balanced"
    ))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "model/loan_approval_model.pkl")
print("Model trained successfully.")
print("Saved to: model/loan_approval_model.pkl")
