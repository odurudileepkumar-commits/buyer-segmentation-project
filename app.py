import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title("Buyer Segmentation Dashboard")

# Load dataset
clients = pd.read_csv("clients.csv")

# Sidebar
st.sidebar.header("Filter")

selected_country = st.sidebar.selectbox(
    "Select Country",
    clients['country'].unique(),
    key="country_filter"
)

# Filter data
filtered_data = clients[clients['country'] == selected_country]

# Dataset Preview
st.subheader("Filtered Dataset")

st.write(filtered_data.head())

# Referral Analysis
st.subheader("Referral Channel Analysis")

referral_count = filtered_data['referral_channel'].value_counts()

fig, ax = plt.subplots()

referral_count.plot(kind='bar', ax=ax)

plt.xlabel("Referral Channel")

plt.ylabel("Count")

st.pyplot(fig)

# Satisfaction Analysis
st.subheader("Satisfaction Score Analysis")

satisfaction_count = filtered_data['satisfaction_score'].value_counts()

fig2, ax2 = plt.subplots()

satisfaction_count.plot(kind='bar', ax=ax2)

plt.xlabel("Satisfaction Score")

plt.ylabel("Count")

st.pyplot(fig2)

# Loan Applied Analysis
st.subheader("Loan Applied Analysis")

loan_count = filtered_data['loan_applied'].value_counts()

fig3, ax3 = plt.subplots()

loan_count.plot(kind='bar', ax=ax3)

plt.xlabel("Loan Applied")

plt.ylabel("Count")

st.pyplot(fig3)

# Pie Chart Analysis
st.subheader("Referral Channel Percentage")

fig4, ax4 = plt.subplots()

referral_count.plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=ax4
)

plt.ylabel("")

st.pyplot(fig4)

# Machine Learning Prediction

st.subheader("Loan Prediction Model")

from sklearn.preprocessing import LabelEncoder

# Create encoder
encoder = LabelEncoder()

# Copy dataset
ml_data = clients.copy()

# Convert text columns to numbers
ml_data['gender'] = encoder.fit_transform(ml_data['gender'])

ml_data['acquisition_purpose'] = encoder.fit_transform(
    ml_data['acquisition_purpose']
)

ml_data['client_type'] = encoder.fit_transform(
    ml_data['client_type']
)

# Features
X = ml_data[['client_type', 'gender', 'acquisition_purpose', 'satisfaction_score']]

# Target
y = ml_data['loan_applied']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Show accuracy
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")
