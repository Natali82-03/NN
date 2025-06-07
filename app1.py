import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Загрузка данных
@st.cache_data
def load_data():
    columns = [
        "age", "gender", "education", "income", "dependents", "housing",
        "monthly_rent", "loan_purpose", "interest_rate", "dti_ratio",
        "loan_term", "credit_score", "loan_approved", "zip_code"
    ]
    df = pd.read_csv("L_Score.csv", names=columns)
    df = df.drop(columns=["zip_code"])
    df["loan_approved"] = df["loan_approved"].map({"Yes": 1, "No": 0})
    return df

# Обучение модели
@st.cache_resource
def train_model(df):
    X = df.drop("loan_approved", axis=1)
    y = df["loan_approved"]

    categorical_features = ["gender", "education", "housing", "loan_purpose"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, acc

# Интерфейс
st.title("✅ Кредитный скоринг")
st.subheader("Введите данные клиента")

df = load_data()
model, acc = train_model(df)

with st.form("form"):
    age = st.number_input("Возраст", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Пол", options=["male", "female"])
    education = st.selectbox("Образование", options=["High School", "Associate", "Bachelor", "Master", "Doctorate"])
    income = st.number_input("Доход ($)", min_value=0, value=50000)
    dependents = st.number_input("Количество иждивенцев", min_value=0, value=0)
    housing = st.selectbox("Тип жилья", options=["OWN", "MORTGAGE", "RENT"])
    monthly_rent = st.number_input("Ежемесячный платеж ($)", min_value=0, value=1500)
    loan_purpose = st.selectbox("Цель кредита", options=[
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", 
        "MEDICAL", "PERSONAL", "VENTURE"
    ])
    credit_score = st.number_input("Кредитный рейтинг", min_value=300, max_value=850, value=650)

    submitted = st.form_submit_button("Проверить одобрение")

if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "education": education,
        "income": income,
        "dependents": dependents,
        "housing": housing,
        "monthly_rent": monthly_rent,
        "loan_purpose": loan_purpose,
        "credit_score": credit_score,
        "interest_rate": 10.0,
        "dti_ratio": 0.2,
        "loan_term": 3.0
    }])
    
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Кредит одобрен!")
    else:
        st.error("❌ Кредит не одобрен.")
