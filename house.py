import streamlit as st
import pickle
import numpy as np

# Load the saved model
model= pickle.load(open(r'C:\Users\DELL\Desktop\VS_code\Machine learning\house_price_model.pkl','rb'))


# Set the title of the Streamlit app
st.title("HOUSE PRICE PREDICTION APP")


# Add a brief description
st.write("This app predicts the price based on living house square feet using a simple linear regression model.")

# Add input widget for user to enter years of experience
price_sqrft = st.number_input("ENTER PRICE($) OF SQRFT:", min_value=0.0, max_value=4500.0, value=1.0, step=0.5)

# When the button is clicked, make predictions
if st.button("Predict House Price"):
    # Make a prediction using the trained model
    sqrft_input = np.array([[price_sqrft]])  # Convert the input to a 2D array for prediction
    prediction = model.predict(sqrft_input)
   
    # Display the result
    st.success(f"The predicted House Price for {price_sqrft} price of sqrft is: ${prediction[0]:,.2f}")
   
# Display information about the model
st.write("The model was trained using a dataset of House data and Price of square feet.built model by Priyanka")
