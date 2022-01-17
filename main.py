import streamlit as st
import pandas as pd
from salary_predict import predict_page
from explore import explore_page
from about import about_page

string = "Knowledge"

st.title("Data Science Project")
st.write("B5.3 Business Software")
page = st.sidebar.selectbox("Pages", ("Salary Prediction", "Explore", "About"))

if page == "Salary Prediction":
    predict_page()
elif page == "Explore":
    explore_page()
else:
    about_page()



    