import streamlit as st
import time

st.title('Real-Time Updates')

progress = st.progress(0)
with st.spinner('Loading...'):
    for i in range(100):
        time.sleep(0.1)
        progress.progress(i + 1)



