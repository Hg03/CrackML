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
    
with st.expander('Q6 Pick an algorithm you like and walk through its math and them implementation of it with its pseudo code ?'):
    mdlit('> Here, you can talk about any particular algorithm that you have worked on and/or feel comfortable discussing.')

with st.expander('Q7 Do you know any tools for running a Machine Learning algorithm in parallel?'):
    mdlit("> Some of the tools, software or hardware, used to execute the Machine Learning algorithms in parallel include Matlab Parfor, GPUs, MapReduce, Spark, Graphlab, Giraph, Vowpal, Parameter Server etc.")
    
with st.expander('Q8 What are the different Machine Learning Approaches ?'):
    mdlit("> The different types of Machine Learning Approaches are :- ")
    mdlit("> **Supervised Learning:** where the output variable (the one you want to predict) is labeled in the training dataset (data used to build the Machine Learning model). Techniques include Decision Trees, Random Forests, Support Vector Machines, Bayesian Classifier etc. For instance, predicting whether a given email is SPAM or not, given sample emails with the labels whether they are SPAM or not, falls within Supervised learning.")
    mdlit("> **Unsupervised Learning:** where the training dataset does not contain the output variable. The objective is to group the similar data together instead of predicting any specific value. Clustering, Dimensionality Reduction and Anomaly Detection are some of the Unsupervised Learning techniques. For instance, grouping the customers based on their purchasing pattern.")
    mdlit("> **Semi-supervised Learning:** This technique falls in between Supervised and Unsupervised Learning because it has a small amount of labeled data with a relatively large amount of unlabeled data. You can find its applications in problems such as Web Content Classification, and Speech Recognition, where it is very hard to get labeled data but you can easily get lots of unlabeled data.")
    mdlit("> **Reinforcement Learning:** Unlike traditional Machine Learning techniques, Reinforcement Learning focuses on finding a balance between Exploration (of unknown new territory) and Exploitation (of current knowledge). It monitors the response of the actions taken through trial and error and measures the response against a reward. The goal is to take such actions for the new data so that the long-term reward is maximized. Let’s say that you are in an unknown terrain, and each time you step on a rock, you get negative reward whereas each time you find a coin, you get a positive reward. In traditional Machine Learning, at each step, you would greedily take such an action whose immediatereward is maximum even though there might be another path for which the overall reward is more. In Reinforcement Learning, after every few steps, you take a less greedy step to explore the full terrain. After much exploration and exploitation, you would know the best way to walk through the terrain so as to maximize your total reward.")

with st.expander('Q9 How would you differentiate Supervised learning and Unsupervised learning ?'):
    mdlit("> **Supervised Learning** is where you have both the input variable x and the output variable y and you use an algorithm to learn the mapping function from x to y and predict the output of the new data. Supervised Learning can further be classified as a Classification or a Regression technique.<br><br> **Unsupervised Learning**, on the other hand, is where you only have the input variable x but no corresponding output variable y. The goal in Unsupervised Learning is to model the underlying structure and distribution of the data. Unsupervised Learning techniques include Clustering, Anomaly Detection, and Dimensionality Reduction.")
    
with st.expander('Q10 What are the different stages to learn hypothesis or models in Machine Learning ?'):
    mdlit("> A hypothesis is a function that is (very close to) the true function which maps the input to the output. The goal is to find such a hypothesis which can learn the true function as efficiently as possible. Following are the three main stages of learning a model:")
    mdlit("> **A. Model Building :** Model building: Learning from the training dataset and building a Machine Learning model using it. ")
    mdlit("> **B. Model Testing:** Testing the learned model using testing dataset.")
    mdlit("> **C Applying the Model:** Model building and testing are performed iteratively until the learned model reaches the desired accuracy. Once the model is finalized, it is applied to the new data.")
    st.code("""
            # Let's use Support Vector Machine for our question.
            from sklearn.svm import SVC
            from sklearn import datasets
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            # In this example, we will use the standard iris dataset available
            iris = datasets.load_iris()
            # Here, we will split it into training and test dataset (90-10 ratio).
            X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10)
            # Model building is initializing a Model with the correct set of parameters
            # and fitting our training dataset.
            model = SVC(kernel='linear')
            model.fit(X_train, y_train)
            # Model testing is predicting the values for test dataset
            y_predicted = model.predict(X_test)
            print(classification_report(y_test, y_predicted))
            # Based on the model's metrics, you can either deploy your model or re-train it.
            """)

with st.expander('Q11 What is the difference between Causation and Correlation ?'):
    mdlit("> **Causation** is a relationship between two variables such that one of them is caused by the occurrence of the other.<br><br>**Correlation**, on the other hand, is a relationship observed between two variables which are related to each other but not caused by one another.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(" ")
    with c2:
        st.image('assets/img2.png')
    with c3:
        st.write(" ")
    mdlit("> In above Figure, you can see that inflation causes the price fluctuations in petrol and groceries so, inflation has a causation relationship with both of them. Between petrol and groceries, there is a correlation that both of them can increase or decrease due to the changes in inflation, but neither of them causes or impacts the other one.")
    
with st.expander('Q12 What is the difference between Online and Offline(batch) learning ?'):
    mdlit("> The major difference is that in case of Online learning, the data becomes available in real-time in a sequential manner, one example at a time, whereas in Offline learning, the complete dataset is statically available. An example of Online learning is a Real-Time recommendation system on amazon.com where Amazon learns from each purchase you make and recommends you similar products.<br><br> Each one has its own advantages and disadvantages. Online learning is time critical so you may not be able to use all the data to train your model whereas with offline learning, you won’t be able to learn in real-time. Quite often, companies use a hybrid approach in which they train the models both online and offline. They would learn a model offline from the static data to interpret global patterns and then incorporate the real-time data for online learning.<br><br> For instance, Twitter could learn a model offline to analyze the sentiments on a global scale. And if an event is happening at a particular place, it could use an online learning model on top of the already learned model to interpret real- time sentiments of the event.")

with st.expander('Q13 What is the difference between Classification and Regression ?'):
    mdlit("> Classification is a kind of Supervised Learning technique where the output label is discrete or categorical. Regression, on the other hand, is a Supervised Learning technique which is used to predict continuous or real-valued variables.<br><br> For instance, predicting stock price is a Regression problem because the stock price is a continuous variable which can take any real-value whereas predicting whether the email is spam or not is a Classification problembecause in this case, the output is discrete and has only two possible values, yes or no.")
    
with st.expander('Q14 What is Sampling and why do we need it ?'):
    mdlit("> Sampling is a process of choosing a subset from a target population which would serve as its representative. We use the data from the sample to understand the pattern in the population as a whole. Sampling is necessary because often we can not gather or process the complete data within a reasonable time. There are many ways to perform sampling. Some of the most commonly used techniques are Random Sampling, Stratified Sampling, and Clustering Sampling.")
    
with st.expander('Q15 What is Stratified Sampling ?'):
    mdlit("> Stratified sampling is a probability sampling technique wherein the entire population is divided into different subgroups called strata, and then aprobability sample is drawn proportionally from each stratum.<br> For instance, in case of a binary classification, if the ratio of positive and negative labeled data was 9:1, then in stratified sampling, you would randomly select subsample from each of the positive and negative labeled dataset such that after sampling, their ratio is still 9:1.<br> Stratified sampling has several advantages over simple random sampling. For example, using stratified sampling, it may be possible to increase the precision with the same sample size or to reduce the sample size required toachieve the same precision.") 
    
with st.expander('Q16 Define Confidence Interval ?'):
    mdlit("> A confidence interval is an interval estimate which is likely to include an unknown population parameter, the estimated range being calculated from the given sample dataset. It simply means the range of values for which you are completely sure that the true value of your variable would lie in.")
    
with st.expander('Q17 What do you mean by i.i.d assumption ?'):
    mdlit("> We often assume that the instances in the training dataset are independent and identically distributed (i.i.d.), i.e, they are mutually independent of each other and follow the same probability distribution. It means that the order in which the training instances are supplied should not affect your model and that the instances are not related to each other. If the instances do not follow an identical distribution, it would be fairly difficult to interpret the data.")
    
with st.expander('Q18 What do we call it Generalized Linear Model (GLM) and when it is clearly nonlinear ?'):
    mdlit("> The Generalized Linear Model (GLM) is a generalization of ordinary linear regression in which the response variables have error distribution models other than a normal distribution. The **linear** component in GLM means that the predictor is a linear combination of the parameters, and it is related to the response variable via a link function. Let us assume that **Y** is the response variable and **X** is the input independent variable. Then,")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(" ")
    with c2:
        st.image('assets/img3.png')
    with c3:
        st.write(" ")
    mdlit("> where E(Y) is the expected value of Y, Xβ is the linear predictor, a linear combination of unknown parameters β and g is the link function.")
    
with st.expander('Q19. Define Conditional Probability ?'):
    mdlit("Conditional Probability is a measure of the probability of one event, given that another event has occurred. Let’s say that you have 2 events, A and B.Then, the conditional probability of A, given B has already occurred, is given as:")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write(" ")
    with c2:
        st.image('assets/img4.png')
    with c3:
        st.write(" ")
        
    mdlit("where ∩ stands for intersection. So, the conditional probability is the joint probability of both the events divided by the probability of event B.")
    
with st.expander('Q20. Are you familiar with Bayes Theorem and why it is useful ?'):
    mdlit("Bayes Theorem is used to describe the probability of an event, based on the prior knowledge of other events related to it. For example, the probability of a person having a particular disease would be based on the symptoms shown.<br>Bayes Theorem can be mathematically formulated as:")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.write(" ")
    with c2:
        st.image('assets/img5.png')
    with c3:
        st.write(" ")
    mdlit("where A and B are the events and P(B) ≠ 0. Most of the times, we want P(A | B) but we know P(B | A). Bayes Theorem is extremely useful in these scenarios, as you can use it to predict P(A | B) using the above equation.<br><br>For instance, let us say that you want to find the probability of a person suffering from liver disease given that he is an alcoholic. Now finding this directly is hard but you can have records of a person being an alcoholic, given that he is suffering from liver disease.<br>Let A be the event that the person has a liver disease and B be the event that he is an alcoholic. You want to find P(A | B), but it is easier to find P(B | A) since it is more common. This is where you can make use of the Bayes Theorem to achieve the desired result.")
    
with st.expander('Q21. How can you get an unbiased estimate of the accuracy of the learned model ?'):
    mdlit("Divide the input dataset into training and test datasets. Build the model using the training dataset and measure its accuracy on the test dataset. For better results, you can use Cross-validation to run multiple iterations of partitioning the dataset into the training and test datasets, analyze the accuracy of the learned model in each iteration and finally use the best model from the learned models.<br>Evaluating the model’s performance with the training dataset is not a good measure because it can easily generate overfitted models which fit well on the given training dataset but do not show similar accuracy on the test dataset. Remember, the goal is to learn a model which can perform well on the new dataset.<br>You can also split the training set into training and validation set and use the validation dataset to prevent the model from overfitting the training dataset.")
    
with st.expander('Q22. How would you handle the scenario where your dataset has missing or dirty (garbled) values?'):
    mdlit("These kind of situations are very common in real life. Sometimes, the data is missing or empty. And sometimes, it can have some unexpected values such as special characters while performing data transformations or saving/fetching the data from the client/server. Another case could be when you expect an ASCII string but receive a Unicode string which may result in garbled data in your string.<br> You can either drop those rows or columns or replace the missing/garbled values with other values such as the mean value of that column or the most occurring value etc. The latter case is generally known as **Parameter Estimation**. Expectation Maximization (EM) is one of the algorithms used for the same. It is an iterative method to find the maximum likelihood or maximum a posteriori (MAP) estimates of the parameters in statistical models.")
