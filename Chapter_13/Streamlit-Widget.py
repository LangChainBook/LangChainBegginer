import streamlit as st

st.title('Interactive Widgets')

name = st.text_input('Enter your name:')
age = st.slider('Select your age:', 0, 100)

st.write(f'Hello {name}, you are {age} years old.')
