import streamlit as st
import pickle

# Load the model from disk
with open('D:/X/AI/Internships/Prodigy InfoTech/Tasks/1 House Price Prediction/model.h5', 'rb') as file:
    model = pickle.load(file)

def predict_sale_price(area, bedrooms, bathrooms):
    # Prepare the input data as a 2D array (since the model expects a 2D array)
    input_data = [[area, bedrooms, bathrooms]]
    
    # Make the prediction
    predicted_price = model.predict(input_data)
    
    return predicted_price[0]

# Streamlit app
st.title('House Sale Price Predictor')

# Get user input
area = st.number_input("Enter the area:", min_value=0, step=1)
bedrooms = st.number_input("Enter the number of bedrooms:", min_value=0, step=1)
bathrooms = st.number_input("Enter the number of bathrooms:", min_value=0, step=1)

if st.button('Predict Sale Price'):
    # Predict the sale price
    predicted_price = predict_sale_price(area, bedrooms, bathrooms)
    st.write(f"The predicted sale price is: ${predicted_price:.2f}")
