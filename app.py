import pandas as pd
import streamlit as st
import numpy as np
import pickle

#data = pd.read_csv('customers.csv')



#st.write(data)
st.title('welcome to Model Deployement page')

model = pickle.load(open('lr_model.pkl','rb'))

feature1 = st.sidebar.number_input("Enter Feature 1")
feature2 = st.sidebar.number_input("Enter Feature 2")
feature3 = st.sidebar.number_input("Enter Feature 3")
feature4 = st.sidebar.number_input("Enter Feature 4")
feature5 = st.sidebar.number_input("Enter Feature 5")

input_data = np.array([feature1,feature2,feature3,feature4,feature5])
input_data = input_data.reshape(1, -1)

st.write('Prediction')
if st.button('predict'):
    prediction = model.predict(input_data)
    st.write(f"The price {prediction[0]}")




