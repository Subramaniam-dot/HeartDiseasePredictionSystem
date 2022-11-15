import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import streamlit as st


heart = pd.read_csv('/home/subramaniam/Projects/Heart disease prediction/heart_disease_data.csv')

#print('dataset:',heart)
#print('no of rows and columns:',heart.shape)
#print(heart.info())

#print('No of data which has null: ',heart.isnull().sum())
#print(heart.describe())
#print(heart['target'].value_counts()) #  Has almost equal distribution.


#X = heart.drop(columns = 'target', axis =1)
#Y = heart['target']

#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify = Y, random_state = 3)

#model = LogisticRegression()
#model.fit(X_train,Y_train)

#X_train_pred = model.predict(X_train)
#training_data_accuracy = accuracy_score(X_train_pred, Y_train)

#print('Accuracy score on Training data: ',training_data_accuracy)

#X_test_pred = model.predict(X_test)
#test_data_accuracy = accuracy_score(X_test_pred,Y_test)
#print('Accuracy score on Test data: ',test_data_accuracy)
#filename='trained_model.sav'
#pickle.dump(model,open(filename,'wb')) #wb= writebinary

loaded_model = pickle.load(open('trained_model.sav','rb'))

def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def HD_pred(input_data):

    

    prediction = loaded_model.predict(input_data)
    return prediction

def main():
    """Heart Disease Prediction System"""
    st.title("Heart Disease Predictive System")
    st.subheader("Machine Learning ")
    
    st.text('--> Below Dataset is used to train the logistic regression Machine Learning Model and helps us predict a person has heart disease or not.')
    #st.text('--> If the qualiy of wine is said to be 7 and above, then it is a Good Quality wine. If not, it is a Bad Quality Wine')
    
    st.write(heart)


    
    activities = ["Prediction",]

    choice = st.sidebar.selectbox("Select",activities)

    if choice == 'Prediction':
        st.info('Prediction with ML')
#age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
        #news_text = st.text_area("Enter Text","Type Here")
        age = st.number_input('Enter age',value=50)
        sex = st.radio("Enter sex, 1-> male, 0-> female",(1, 0))
        #sex =  st.number_input('Enter sex, 1-> male, 0-> female',step=1e-6,format="%.5f")
        cp =  st.radio("Enter sex, 1-> male, 0-> female",(0,1,2,3))


        trestbps= st.number_input('Enter resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))',value = 1)
        chol = st.number_input('Enter serum cholesterol in mg/dl',value = 1)


        fbs = st.radio("Enter if fasting blood sugar > 120 mg/dl, 1->True, 0-> False",(1, 0))

        restecg = st.radio("Enter ecg observation at resting condition, 1-> normal, 0-> lv hypertrophy, 2 -> others",(0,1,2))

        thalach = st.number_input('Enter maximum heart rate achieved',value = 1)


        exang = st.radio("Enter exercise-induced angina (True/ False), 1->True, 0-> False",(1, 0))

        oldpeak = st.number_input('Enter ST depression induced by exercise relative to rest',step=0.1,format="%.1f")
        slope = st.radio("Enter the slope of the peak exercise ST segment,2->upsloping  ,1->Flat , 0->downsloping ",(0,1,2))
        
        ca = st.radio("Enter number of major vessels (0-3) colored by fluoroscopy",(0,1,2,3))
        thal = st.radio("Enter normal, fixed defect, reversible defect ,0-> others,1-> fixed defect, 2-> Normal, 3-> reversable defect",(0,1,2,3))

    

        input_data=''

        input_data = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

# changing the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)

#reshape the data as we are the predicting the label for only one instance

        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
       


#fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol
        all_ml_model = ['Logistic Regression']
        model_choice = st.selectbox('Choose model',all_ml_model)
        prediction_labels = {'This person does not have a heart disease':0 , 'This person have a heart disease':1}
        if st.button('classify'):
            
            if model_choice == 'Logistic Regression':
                
                result = HD_pred(input_data_reshaped)
                #prediction = loaded_model.predict(input_data_reshaped)
                st.write(result)
                final_results = get_keys(result,prediction_labels)
                st.success(final_results)
           
    
    
    
    st.sidebar.header('About Data set:')

    st.sidebar.text('This dataset is available from the UCI machine learning repository, https://archive.ics.uci.edu/ml/datasets/heart+disease.')

    st.sidebar.text('Citations:')
    st.sidebar.text('1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D')
    st.sidebar.text('2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D')
    st.sidebar.text('3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D')
    st.sidebar.text('4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:Robert Detrano, M.D., Ph.D')

      
    st.sidebar.write('Done By [Subramaniam](https://subramaniam-dot.github.io)')





if __name__ =='__main__':
 main()