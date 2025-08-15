# Fraud Transaction Detector

A machine learning application for detecting potentially fraudulent financial transactions using an **Isolation Forest** anomaly detection model.

The system is designed to process transaction data, apply preprocessing using a stored scaler, and predict whether each transaction is normal or potentially fraudulent. It is lightweight, fast, and suitable for integration into real-time or batch-processing financial pipelines.

---

## 📌 Table of Contents
1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Input Format](#input-format)  
7. [Output](#output)  
8. [Project Structure](#project-structure)  
9. [Troubleshooting](#troubleshooting)  
10. [License](#license)  

---

## 📝 Introduction
The **Fraud Transaction Detector** uses an **Isolation Forest** model to identify anomalies in transaction data.  
Anomalies are flagged as potential fraud based on how different they appear compared to the majority of transactions.

The project includes:
- A pre-trained Isolation Forest model (`iso_forest_model.pkl`)
- A pre-trained scaler (`scaler.pkl`) to normalize transaction features before prediction

---

## 🚀 Features
- **Anomaly Detection**: Detects suspicious transactions in real-time or from batch files
- **Preprocessing Pipeline**: Scales and transforms features to match training distribution
- **Lightweight & Fast**: Suitable for low-latency applications
- **Easy Integration**: Can be embedded in APIs or transaction processing systems

---

## 📦 Requirements
Python version:
```
Python 3.8 or later
```

Dependencies (from `requirements.txt`):
```
scikit-learn
numpy
pandas
joblib
flask          # (if using API mode)
```

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/valive421/fraud_transaction_detector.git
cd fraud_transaction_detector
```

Install dependencies:
```bash
pip install -r requirements.txt
```

(Optional) Run inside a development container:
```bash
# Build and run with Docker
docker build -t fraud-detector .
docker run -it fraud-detector
```

---

## 🖥️ Usage

### 1️⃣ Command-Line Mode
```bash
python app.py --input sample_transactions.csv
```
- `--input`: Path to a CSV file containing transaction records.

### 2️⃣ API Mode (if implemented)
Start the API server:
```bash
python app.py --mode api --host 0.0.0.0 --port 5000
```
Send a request:
```bash
curl -X POST http://localhost:5000/predict      -H "Content-Type: application/json"      -d '{"amount": 150.75, "time_delta": 120, "transaction_type": 1, "balance_before": 2000, "balance_after": 1849.25}'
```

---

## 📥 Input Format
The model expects numerical features in the same order used during training.  
Example CSV format:
```csv
amount,time_delta,transaction_type,balance_before,balance_after
150.75,120,1,2000,1849.25
42.50,30,0,150,107.50
```

**Columns**:
| Feature            | Type  | Description                                |
|--------------------|-------|--------------------------------------------|
| `amount`           | float | Transaction amount                         |
| `time_delta`       | float | Time since last transaction (seconds)      |
| `transaction_type` | int   | Encoded transaction type                   |
| `balance_before`   | float | Account balance before transaction         |
| `balance_after`    | float | Account balance after transaction          |

---

## 📤 Output
Predictions are:
- `1` → Non-fraudulent transaction
- `-1` → Potentially fraudulent transaction

Example:
```bash
Transaction ID 1: -1 (Potential Fraud)
Transaction ID 2: 1  (Normal)
```

If anomaly scores are enabled, output may look like:
```json
{
  "prediction": -1,
  "score": -0.15
}
```

---

## 📂 Project Structure
```
fraud_transaction_detector/
│
├── app.py                 # Main application script
├── iso_forest_model.pkl   # Pre-trained Isolation Forest model
├── scaler.pkl             # Preprocessing scaler
├── requirements.txt       # Project dependencies
├── .devcontainer/         # Dev container configs
└── README.md              # Documentation
```

---

## 🛠️ Troubleshooting
**Common Issues:**
- **ValueError: Feature mismatch** → Ensure input features match training order and count  
- **FileNotFoundError: 'iso_forest_model.pkl'** → Verify model file is present in project root  
- **ImportError** → Install dependencies using `pip install -r requirements.txt`  

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
