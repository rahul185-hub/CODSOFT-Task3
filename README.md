# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn (leave the service) or continue using it. The project also includes a Streamlit web application for real-time predictions.

---

## Project Overview

The objective of this project is to predict customer churn using Machine Learning. Users can enter customer details through a Streamlit web interface, and the model predicts whether the customer is likely to churn or not.

---

## Dataset

This project uses the **Customer Churn Dataset** from Kaggle.

Dataset Link:
https://www.kaggle.com/datasets/abdulwadood11220/customer-churn-dataset

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Support Vector Machine (SVM)
- GridSearchCV
- StandardScaler
- Joblib
- Streamlit

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── app.py                     # Streamlit web application
├── notebook.py                # Machine learning training code
├── customer_churn_data.csv    # Dataset
├── model.pkl                  # Trained machine learning model
├── scaler.pkl                 # Saved StandardScaler
├── requirements.txt
└── README.md
```

---

## How It Works

1. Load the customer churn dataset.
2. Preprocess the data.
3. Encode categorical values.
4. Split the dataset into training and testing sets.
5. Scale the input features.
6. Train the Support Vector Machine (SVM) model.
7. Save the trained model using Joblib.
8. Load the model in the Streamlit application.
9. Enter customer details and get a churn prediction.

---

## Features Used

- Age
- Gender
- Tenure
- Monthly Charges

---

## Model Used

- Support Vector Machine (SVM)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rahul185-hub/CODSOFT-Task3.git
```

Move into the project folder:

```bash
cd CODSOFT-Task3
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

---

## Screenshots

### Home Page

![Home Page](Screenshot%202026-07-31%20102633.png)

### Prediction Result

![Prediction Result](Screenshot%202026-07-31%20102532.png)

---

## Author

**Rahul**

GitHub:
https://github.com/rahul185-hub

---

## License

This project is created for learning purposes as part of the **CodSoft Machine Learning Internship**.
