# -*- coding: utf-8 -*-

'''
streamlit run D:/gitbuild/bosch/attrition.py
'''

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.metrics import classification_report as cr
import os,time

# ============
# page design
# ============
st.set_page_config(layout='wide')

if "hrdata" not in st.session_state:
    st.session_state["concdata"] = None
    
if "predicted" not in st.session_state:
    st.session_state["predicted"] = False

if "predictions" not in st.session_state:
    st.session_state["predictions"] = None
    
if "err" not in st.session_state:
    st.session_state["err"] = -1

if "model" not in st.session_state:
    st.session_state["model"] = None

def homepage():
    st.header("Demo : Classification")
    st.divider()
    st.subheader("Employee Attrition Prediction")

def dataset():
    st.session_state["predicted"] = False
    
    st.header("Dataset")
    # file="D:/gitbuild/bosch/hr_emp.csv"
    file = "hr_emp.csv"
    
    data = pd.read_csv(file)
    data['attr']=0
    data.attr[data.attrition=='Yes'] = 1
    data.drop(columns=['attrition'],inplace=True)

    st.dataframe(data)

    tot = len(data)
    cols = len(data.columns) - 1
    
    st.success("Total Records = " + str(tot))
    st.success("Total Features = " + str(cols))
    st.session_state["hrdata"] = data


def predictattr():
    churn = st.session_state["hrdata"]
    
    with st.spinner("Building Regression Model and Predicting ..."):
        
        fact_x = churn.select_dtypes(include=['object']).columns
    
        # update a few strings for easy reading
        churn.travel[churn.travel=='Travel_Rarely']='Rarely'
        churn.travel[churn.travel=='Travel_Frequently']='Frequently'
        churn.travel[churn.travel=='Non-Travel']='No'
        churn.travel.unique()
    
        churn.department[churn.department=='Research & Development']='R&D'
        churn.department[churn.department=='Human Resources']='HR'
        churn.department.unique()
    
        # make a copy of the dataset
        newchurn=churn.copy(deep=True)
    
        # convert columns to dummies
        for e in fact_x:
            dummy=pd.get_dummies(churn[e],drop_first=True,prefix=e)
            newchurn = newchurn.join(dummy)
    
        newchurn.drop(columns=fact_x,inplace=True)
        
        trainx,testx,trainy,testy = train_test_split(newchurn.drop('attr',axis=1),newchurn['attr'],test_size=0.2)
    
        # build the model
        m1=sm.Logit(trainy,trainx).fit()
    
        # predictions
        p1=m1.predict(testx)
    
        # convert probabilities to 0 and 1
        predY = p1.copy()
        predY[predY < 0.3]=0
        predY[predY >=0.3]=1
    
        res = pd.DataFrame({'actual':testy,'predicted':predY})
        time.sleep(2)
        
        c1,c2 = st.columns(2)
        c1.header("Confusion Matrix")
        cm = pd.crosstab(res.actual,res.predicted,margins=True)
        c1.write(cm)
        
        c2.header("Classification Report")
        crep = cr(res.actual,res.predicted,output_dict=True)
        df_crep = pd.DataFrame(crep).T
        c2.write(df_crep)

    

# ==============================================
# calling each function based on the click value
# ==============================================
# main menu settings
options=[":house:",":memo:",":lower_left_fountain_pen:"] 
captions=['Home','Dataset',"Attrition Prediction"]
nav = st.sidebar.radio("Select Option",options,captions=captions)
ndx = options.index(nav)

if (ndx==0):
    homepage()

if (ndx==1):
    dataset()
    
if (ndx==2):
    predictattr()
