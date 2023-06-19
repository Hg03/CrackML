import streamlit as st
from markdownlit import mdlit
from pdf_mail import sendpdf

st.set_page_config(layout='wide')

def send_pdf(sender_email_address,receiver_email_address,password,subject,body,filename,location_of_file):
   
    filename = input()        
    
    # ex-"C:/Users / Vasu Gupta/ "
    location_of_file = 'book/Cracking The Machine Learning Interview.pdf'
    
    
    # Create an object of sendpdf function 
    k = sendpdf(sender_email_address, receiver_email_address,password,subject,body,filename,location_of_file)
    
    # sending an email
    k.email_send()

mdlit('# [blue]Cracking[/blue] the [yellow]Machine Learning[/yellow] Interview 🔥')

st.sidebar.title('Why you are here')
with st.form('My form'):
    email_id = st.sidebar.text_input('Your Email ID')
    submit = st.sidebar.button('Submit')
    if submit:
        send_pdf('gehloth03@gmail.com',email_id,st.secrets['password'],'Cracking The Machine Learning Interview',"Here's your pdf",'Cracking The Machine Learning Interview.pdf','book/Cracking The Machine Learning Interview.pdf')
        st.sidebar.success('PDF sent successfully !!')

with st.expander('Q1. Define Machine Learning and How it is different from Artificial Intelligence ?'):
    mdlit('> Machine Learning is a subset of Artificial Intelligence that aims at making systems learn automatically from the data provided and improve their learnings over time without being explicitly programmed. Artificial Intelligence (AI) is the broader concept of machines being able to carry out tasks in a way that could be considered as smart. The machines not necessarily learn from the data but may exhibit intelligence in performing certain tasks that mimic the characteristic of human intelligence. Above and beyond Machine Learning, AI includes Self Driving Cars, Natural Language Processing (NLP), Knowledge Representation etc. As can be seen in below figure, <br><br> ')
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write(' ')
    with c2:
        st.image('assets/img1.png')
    with c3:
        st.write(' ')
    mdlit('> Deep learning is a subset of Machine Learning which itself is a subset of the overall Artificial Intelligence concept.')
    
with st.expander('Q2. How would you differentiate a Machine Learning Algorithm from other algorithms ?'):
    mdlit('> A Machine Learning algorithm is an application that can learn from the data without relying on the explicit instructions to follow. On the contrary, a traditional algorithm is a process or set of rules to be followed especially by a computer, which does not learn anything on its own. For instance, let us say that you want to compute the sum of 2 numbers. A traditional algorithm would implement a sum function which is explicitly programmed to take the input numbers and return their sum. However, a Machine Learning algorithm would take as input the sample (training) dataset with the numbers and their corresponding sum and would learn the pattern automatically such that given a new pair of numbers (test data), it would return their sum itself without being explicitly programmed to do so.<br><br> Lets Train a sckit learn based linear model that returns the sum of two number - <br><br>')
    st.code('''
            from sklearn import linear_model
            import numpy as np
            # First, create our training (input) dataset. The input_data has2 input integers and the input_sum is the resulting sum of them.
            input_data = np.random.randint(50, size=(20, 2))
            input_sum = np.zeros(len(input_data))
            for row in range(len(input_data)):
            input_sum[row] = input_data[row][0] + input_data[row][1]
            # Now, we will build a simple Linear Regression model which trains on this dataset.
            linear_regression_model = linear_model.LinearRegression(fit_intercept=False)
            linear_regression_model.fit(input_data, input_sum)
            # Once, the model is trained, let's see what it predicts for the new data.
            predicted_sum = linear_regression_model.predict([[60, 24]])
            print("Predicted sum of 60 and 24 is " + str(predicted_sum))
            # To give you an insight into this model, it predicts the output usingthe following equation:
            # output = <coefficient for 1st number> * < 1st number> + <coefficient for 2nd number> * < 2nd number>
            # Now, our model should have 1, 1 as the coefficients which means it figured out
            # that for 2 inout integers, it has to return their sum
            print("Coefficients of both the inputs are " + str(linear_regression_model.coef_))
            ''')
    
with st.expander('Q3. What do you understand by Deep Learning and what are some main characteristics that distinguishes it from tradition Machine Learning ?'):
    mdlit('> As shown in above Figure , Deep learning is a subset of the broader Machine Learning concept, and is inspired by the function and the structure of the human brain. **Deep Learning is based on learning the data representation itself**.<br><br>')
    mdlit('''>> **A.**The most important difference between Deep Learning and traditional Machine Learning is the performance of Deep Learning as the scale of data increases. Deep Learning algorithms need a large amount of data to understand it perfectly, which is why they do not perform well when given a small dataset. <br> **B.** Deep Learning algorithms try to learn thehigh-level features (important and relevant characteristics of the dataset) on their own, as opposed to the traditional Machine Learning algorithms, which require a manual input of the feature set.<br> **C.**Deep Learning algorithms rely heavily on GPU and generally need high-end machines to train. This is because they perform a large number of multiplications and other operations which could be highly parallelized in GPUs. A Deep Neural Network consists of many hidden layers with hundreds of neurons in each layer and each layer performs the same computation. Using a high-end machine and/or GPU would drastically speed up the overall processing by executing each layer’s computation in parallel.''')

with st.expander('Q4. What is the difference between Data mining and Machine learning ?'):
    mdlit("> Machine Learning is a branch of Artificial Intelligence which aims at making systems learn automatically from the data provided and improve their learning over time without being explicitly programmed.<br> Data Mining, on the other hand, focuses on analyzing the data and extracting knowledge and/or unknown interesting patterns from it. The goal is to understand the patterns in the data in order to explain some phenomenon and not to develop a sophisticated model which can predict the outcomes for the unknown/new data.<br> For instance, you can use Data Mining on the existing data to understand your company’s sales trends and then build a Machine Learning Model to learn from that data, find the correlations and adapt for the new data.")
    
with st.expander('Q5. What is Inductive Machine Learning ?'):
    mdlit("> Inductive Machine Learning involves the process of learning by examples, where a system tries to induce a general rule from a set of observed instances. The classic Machine Learning approach follows the paradigm of induction and deduction.<br> Inductive Machine Learning is nothing but the inductive step in which you learn the model from the given dataset. Similarly, the deductive step is the one in which the learned model is used to predict the outcome of the test dataset.")
