# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="EEG Seizure Detection", layout="centered")

st.title("🧠 EEG Seizure Detection")
st.subheader("Epileptic Seizure Recognition")

uploaded_file = st.file_uploader("📂 Upload your EEG CSV file", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.write("✅ Data uploaded successfully! Here's a preview:")
        st.dataframe(df.head())

        # Load model and scaler (they were saved together as a tuple)
        model, scaler = joblib.load("model.pkl")

        # Drop label column if exists
        if 'y' in df.columns:
            X = df.drop('y', axis=1)
        else:
            X = df.copy()

        # Preprocess (scale) and predict
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)

        # Display results
        st.subheader("🧪 Prediction Results")
        st.write(prediction)

        # Optional: summarize class counts
        st.write("🔢 Prediction Summary:")
        st.write(pd.Series(prediction).value_counts().rename_axis("Class").reset_index(name="Count"))

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("👈 Please upload a CSV file to start.")
