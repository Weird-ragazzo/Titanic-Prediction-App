import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("Titanic_RegressionModel.pkl")
scaler = joblib.load("TitanicScaler.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details below:")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", 0, 8, 0)
parch = st.number_input("Number of Parents/Children Aboard", 0, 6, 0)
fare = st.number_input("Fare", 0.0, 600.0, 32.2)
embarked_Q = st.selectbox("Embarked at Queenstown?", ["No", "Yes"])
embarked_S = st.selectbox("Embarked at Southampton?", ["No", "Yes"])

# Data preparation
sex = 0 if sex == "male" else 1
embarked_Q = 1 if embarked_Q == "Yes" else 0
embarked_S = 1 if embarked_S == "Yes" else 0

# Scale numerical features
input_df = pd.DataFrame([[age, fare, sibsp, parch]],
                        columns=["Age", "Fare", "SibSp", "Parch"])
input_df[["Age", "Fare", "SibSp", "Parch"]] = scaler.transform(input_df)

# Add categorical features
input_df["Pclass"] = pclass
input_df["Sex"] = sex
input_df["Embarked_Q"] = embarked_Q
input_df["Embarked_S"] = embarked_S

# Reorder columns if needed
input_df = input_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]]

# Prediction button
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("✅ This passenger would have SURVIVED!")
    else:
        st.error("❌ Unfortunately, this passenger would NOT have survived.")
