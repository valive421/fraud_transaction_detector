from gettext import translation
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib  

# Load the trained model and scaler
iso_forest = joblib.load('iso_forest_model.pkl')  
scaler = joblib.load('scaler.pkl') 

def classify_fraudulent_transactions(new_data: pd.DataFrame, model, scaler: StandardScaler) -> pd.DataFrame:
    # Check if data is correct
    if 'Amount' not in new_data.columns or 'Time' not in new_data.columns:
        raise ValueError("The dataset must include 'Amount' and 'Time' columns.")
    
    # Clean the data
    object_columns = new_data.loc[:, new_data.dtypes == "object"].columns
    for col in object_columns:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    new_data.dropna(axis=0, inplace=True)

    # Scale the data as trained
    new_data[['Amount', 'Time']] = scaler.transform(new_data[['Amount', 'Time']])
    # Make predictions
    predictions = model.predict(new_data)
    # Filter fraud transactions
    new_data['Fraudulent'] = [1 if x == -1 else 0 for x in predictions]
    # Return the prediction
    return new_data

st.title('Credit Card Fraud Detection')

uploaded_file = st.file_uploader("Upload an Excel file with credit card transactions", type="xlsx")

if uploaded_file is not None:
    new_data = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:")
    st.write(new_data.head())
    
    try:
        results = classify_fraudulent_transactions(new_data, iso_forest, scaler)
        st.write("Results with Fraudulent Column:")
        st.write(results)
        
        # Visualizations using Streamlit
        st.subheader("Distribution of Transaction Amounts")
        non_fraudulent_amounts = results[results['Fraudulent'] == 0]['Amount']
        fraudulent_amounts = results[results['Fraudulent'] == 1]['Amount']
        st.bar_chart(non_fraudulent_amounts, use_container_width=True, title='Non-Fraudulent Amounts')
        st.bar_chart(fraudulent_amounts, use_container_width=True, title='Fraudulent Amounts')
        
        st.subheader("Distribution of Time")
        non_fraudulent_times = results[results['Fraudulent'] == 0]['Time']
        fraudulent_times = results[results['Fraudulent'] == 1]['Time']
        st.bar_chart(non_fraudulent_times, use_container_width=True, title='Non-Fraudulent Times')
        st.bar_chart(fraudulent_times, use_container_width=True, title='Fraudulent Times')
    
    except Exception as e:
        st.error(f"Error: {e}")
