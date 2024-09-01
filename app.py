import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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

# Set custom style for Seaborn
sns.set(style="whitegrid")

st.title('🔍 Credit Card Fraud Detection')
st.markdown("""
    Detect fraudulent transactions with a trained Isolation Forest model. Upload your data and get insights immediately.
""")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Excel file with credit card transactions", type="xlsx")

if uploaded_file is not None:
    new_data = pd.read_excel(uploaded_file)
    st.write("📄 **Uploaded Data Preview**")
    st.dataframe(new_data.head())
    
    try:
        results = classify_fraudulent_transactions(new_data, iso_forest, scaler)
        st.write("🔍 **Results with Fraudulent Column**")
        st.dataframe(results)

        # Visualizations with enhanced design
        st.subheader("💰 Distribution of Transaction Amounts")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results[results['Fraudulent'] == 0]['Amount'], bins=50, kde=True, label='Non-Fraudulent', color='green', ax=ax)
        sns.histplot(results[results['Fraudulent'] == 1]['Amount'], bins=50, kde=True, label='Fraudulent', color='red', ax=ax)
        ax.set_title('Distribution of Transaction Amounts', fontsize=16)
        ax.set_xlabel('Transaction Amount', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend()
        st.pyplot(fig)
        
        st.subheader("⏰ Distribution of Transaction Time")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(results[results['Fraudulent'] == 0]['Time'], bins=50, kde=True, label='Non-Fraudulent', color='blue', ax=ax)
        sns.histplot(results[results['Fraudulent'] == 1]['Time'], bins=50, kde=True, label='Fraudulent', color='orange', ax=ax)
        ax.set_title('Distribution of Transaction Time', fontsize=16)
        ax.set_xlabel('Transaction Time', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.legend()
        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"⚠️ **Error:** {e}")



        st.pyplot(fig)
    
    except Exception as e:
        st.error(f"Error: {e}")
