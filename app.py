import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Titanic ML Dashboard",
    page_icon="🚢",
    layout="wide"
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ---------------- LOAD MODEL ----------------
model = joblib.load(os.path.join(MODEL_DIR, "titanic_model.pkl"))
gender_encoder = joblib.load(os.path.join(MODEL_DIR, "gender_encoder.pkl"))
port_encoder = joblib.load(os.path.join(MODEL_DIR, "port_encoder.pkl"))

# ---------------- LOAD DATA (for charts) ----------------
df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "titanic_dataset_700_records.csv"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Prediction"])

# ================= DASHBOARD =================
if page == "📊 Dashboard":

    st.title("🚢 Titanic Analytics Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Passengers", len(df))
    col2.metric("Survived", int(df["survived"].sum()))
    col3.metric("Death Rate %", round(100 - df["survived"].mean()*100, 2))

    st.divider()

    # ---------------- Charts ----------------

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Survival Distribution")

        fig1, ax1 = plt.subplots()
        df["survived"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            ax=ax1
        )
        ax1.set_ylabel("")
        st.pyplot(fig1)

    with col2:
        st.subheader("Gender Survival Comparison")

        fig2, ax2 = plt.subplots()
        pd.crosstab(df["gender"], df["survived"]).plot(
            kind="bar",
            ax=ax2
        )
        st.pyplot(fig2)

    st.subheader("Age Distribution")

    fig3, ax3 = plt.subplots()
    df["age"].hist(bins=20, ax=ax3)
    ax3.set_xlabel("Age")
    ax3.set_ylabel("Count")
    st.pyplot(fig3)


# ================= PREDICTION =================
elif page == "🔮 Prediction":

    st.title("🔮 Survival Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        passenger_class = st.selectbox("Passenger Class", [1, 2, 3])
        gender = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 0, 100, 25)
        ticket_fare = st.number_input("Ticket Fare", 0.0, 500.0, 50.0)

    with col2:
        siblings_spouses = st.number_input("Siblings/Spouses", 0, 10, 0)
        parents_children = st.number_input("Parents/Children", 0, 10, 0)
        embarkation_port = st.selectbox("Port", ["S", "C", "Q"])

    gender_encoded = gender_encoder.transform([gender])[0]
    port_encoded = port_encoder.transform([embarkation_port])[0]

    if st.button("🚀 Predict", use_container_width=True):

        input_data = pd.DataFrame([[
            passenger_class,
            gender_encoded,
            age,
            siblings_spouses,
            parents_children,
            ticket_fare,
            port_encoded
        ]], columns=[
            "passenger_class",
            "gender",
            "age",
            "siblings_spouses",
            "parents_children",
            "ticket_fare",
            "embarkation_port"
        ])

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0]

        st.subheader("Result")

        if prediction == 1:
            st.success("🎉 Passenger SURVIVED")
        else:
            st.error("💀 Passenger DID NOT SURVIVE")

        st.write(f"📊 Survival Probability: **{round(prob[1]*100, 2)}%**")
        st.write(f"📊 Death Probability: **{round(prob[0]*100, 2)}%**")
