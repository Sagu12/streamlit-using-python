

import streamlit as st

import pandas as pd

import sklearn

import numpy as np

import matplotlib.pyplot as plt

from sklearn import datasets

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

#used to set the title
st.title('ML APP')

#used to write something as markdown
st.write("""
# Explore different classifiers
which one is the best?
""")

#Adding widgets

dataset_name= st.sidebar.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset"))

st.write(dataset_name)

classifier_name= st.sidebar.selectbox("Select classifier", ("KNN", "Random forest", "SVM"))

st.write(classifier_name)

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data= datasets.load_iris()
        
    elif dataset_name == "Breast Cancer":
        data= datasets.load_breast_cancer()
        
    else:
        data= datasets.load_wine()
        
    X=data.data
    y=data.target
    
    return X,y

X,y= get_dataset(dataset_name)

st.write("Shape of dataset", X.shape)

st.write("Number of classes", len(np.unique(y)))



#Adding parameters

def add_parameters(clf_name):
    params= dict()
    
    if clf_name == "KNN":
        K= st.sidebar.slider("K", 1,15)
        
        params["K"]= K
        
    elif clf_name == "SVM":
        C= st.sidebar.slider("C", 1,10)
        
        params["C"]= C
        
    else:
        max_depth= st.sidebar.slider("max_depth", 2, 15)
        
        n_estimators= st.sidebar.slider("n_estimators", 2, 150)
        
        params["max_depth"]= max_depth
        params["n_estimators"]= n_estimators
        
    return params

params=add_parameters(classifier_name)

#Adding classifiers

def get_classifier(params,clf_name):
    
    if clf_name == "KNN":
        clf= KNeighborsClassifier(n_neighbors=params["K"])
       
        
        
    elif clf_name == "SVM":
        clf= SVC(C=params["C"])
        
        
        
    else:
        clf= RandomForestClassifier(n_estimators=params["n_estimators"], max_depth= params["max_depth"], random_state=1234)
        
    return clf

clf= get_classifier(params,classifier_name)


#Classification part

X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)

y_pred= clf.predict(X_test)

acc= accuracy_score(y_pred,y_test)

st.write("classifier = {}".format(classifier_name))

st.write("accuracy= {}".format(acc))


#PLOTTING THE GRAPH USING PCA TECHNIQUE TO REDUCE THE DIMENSIONS BY REDUCING THE FEATURES

pca= PCA(2)

X_projected= pca.fit_transform(X)

X1= X_projected[:,0]
X2= X_projected[:,1]

fig= plt.figure()

plt.scatter(X1,X2, c=y, alpha=0.8, cmap="viridis")

plt.xlabel("Principple component 1")

plt.ylabel("Principle component 2")

plt.colorbar()

st.pyplot()



        
