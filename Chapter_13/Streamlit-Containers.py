import streamlit as st

st.title('Layouts and Containers')

with st.sidebar:
    st.header('Sidebar')
    st.write('This is the sidebar.')

col1, col2 = st.columns(2)
with col1:
    st.header('Column 1')
    st.write('Content for column 1.')

with col2:
    st.header('Column 2')
    st.write('Content for column 2.')
