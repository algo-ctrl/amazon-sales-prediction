# Amazon Sales Prediction Model

## Project Overview
This project is a Machine Learning application that predicts Amazon product sales based on features such as product type, discount, shipping cost, marketing spend, and other relevant attributes.

It uses a Random Forest Regressor model trained on preprocessed data and provides a web interface built using Streamlit for making real-time predictions.

---

## Features
- Clean data preprocessing and encoding
- Random Forest Regression model for accurate predictions
- StandardScaler for consistent feature scaling
- Streamlit-based web interface for easy user interaction
- Modular structure separating model training and application logic
- Ready-to-deploy setup with requirements and configuration files

---

## Project Structure


AmazonSalesPrediction/
├── model_training.py # Script to train and save the ML model
├── app.py # Streamlit web app for predictions
├── amazon_sales_predictor.pkl # Trained machine learning model
├── scaler.pkl # StandardScaler object used during training
├── label_encoders.pkl # Encoded categorical data mappings
├── requirements.txt # List of dependencies
├── README.md # Project documentation
└── .gitignore # Files and directories to ignore in GitHub



---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AmazonSalesPrediction.git
cd AmazonSalesPrediction

### 2. Install dependencies
pip install -r requirements.txt

### 3. Train the model (if not already trained)
python model_training.py


### 4. Run the Streamlit app
streamlit run app.py

### 5. Open in browser

After running the above command, Streamlit will display a local URL (e.g., http://localhost:8501).
Open it in your browser to access the Amazon Sales Predictor web interface.

### Input Fields (in App)

Product

Category

Customer Location

Payment Method

Order Status

Quantity Ordered

Price per Unit

Day

Month

Weekday




### Output

The app displays the predicted total sales value based on the input details.

### Requirements

The following Python libraries are required:

pandas

numpy

scikit-learn

streamlit

joblib

Install them using:

pip install -r requirements.txt
