import streamlit as st
from markdownlit import mdlit

st.set_page_config(layout='wide')

mdlit('# [blue]Cracking[/blue] the [yellow]Machine Learning[/yellow] Interview 🔥')

st.sidebar.title('Why you are here')
with st.form('My form'):
    email_id = st.sidebar.text_input('Your Email ID')
    submit = st.sidebar.button('Submit')

with st.expander('Q1. Define Machine Learning and How it is different from Artificial Intelligence ?'):
    mdlit('> Machine Learning is a subset of Artificial Intelligence that aims at making systems learn automatically from the data provided and improve their learnings over time without being explicitly programmed. Artificial Intelligence (AI) is the broader concept of machines being able to carry out tasks in a way that could be considered as smart. The machines not necessarily learn from the data but may exhibit intelligence in performing certain tasks that mimic the characteristic of human intelligence.')
