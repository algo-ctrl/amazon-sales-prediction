Amazon Sales Prediction Model
Project Overview

This project is a Machine Learning application that predicts Amazon product sales based on features such as product type, discount, shipping cost, marketing spend, and other relevant attributes.

It uses a Random Forest Regressor model trained on preprocessed data and provides a web interface built using Streamlit for making real-time predictions.

Features

Clean data preprocessing and encoding

Random Forest Regression model for accurate predictions

StandardScaler for consistent feature scaling

Streamlit-based web interface for easy user interaction

Modular structure separating model training and application logic

Ready-to-deploy setup with requirements and configuration files

Project Structure
AmazonSalesPrediction/
├── model_training.py           # Script to train and save the ML model  
├── app.py                      # Streamlit web app for predictions  
├── amazon_sales_predictor.pkl  # Trained machine learning model  
├── scaler.pkl                  # StandardScaler object used during training  
├── requirements.txt            # List of dependencies  
├── README.md                   # Project documentation  
└── .gitignore                  # Files and directories to ignore in GitHub

Installation and Setup
1. Clone the Repository
git clone https://github.com/<your-username>/AmazonSalesPrediction.git
cd AmazonSalesPrediction

2. Install Dependencies
pip install -r requirements.txt

3. Train the Model (if not already trained)
python model_training.py


This will generate two files:

amazon_sales_predictor.pkl

scaler.pkl

4. Run the Web App
streamlit run app.py


The app will open automatically in your browser at:

http://localhost:8501/

Model Information

Algorithm: Random Forest Regressor

Libraries Used: scikit-learn, pandas, numpy

Metrics Evaluated: R² Score, Mean Absolute Error

Data Processing: Label Encoding for categorical variables and StandardScaler for normalization

Example Input
Feature	Example Value
Product Type	Electronics
Category	Laptop
Discount	10
Shipping Cost	50
Marketing Spend	5000
Rating	4.2
Output

The app predicts the estimated sales value for the entered parameters and displays the result immediately on the screen.

Dependencies

All project dependencies are listed in the requirements.txt file.
Main dependencies include:

streamlit

pandas

numpy

scikit-learn

joblib

License

This project is open-source and distributed under the MIT License.
