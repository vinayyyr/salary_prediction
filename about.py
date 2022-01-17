import streamlit as st
import webbrowser


def about_page():
  
  
  st.write("""
			### Salary Prediction Machine Learning Web Application 
			#### Built with Streamlit
			### By
			+ Vinay Raj
			+ rajvinay198@gmail.com
			""")
  url = 'https://github.com/vinayyyr'

  if st.button('Github'):
			webbrowser.open_new_tab(url)
  url = 'https://www.linkedin.com/in/vinayrajj/'
  if st.button('Linkedln'):
			webbrowser.open_new_tab(url)	
  st.markdown("![Salary Prediciton Machine Learning Web Application](https://media.giphy.com/media/3o8dFzIXb0qaE3pYWs/giphy.gif)")	


