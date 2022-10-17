from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import tflearn
from tflearn.data_utils import load_csv

def DatasetPreprocess():
    # This Function for Preprocess the dataset 
    tumor_dataset = pd.read_csv('Tumor Cancer Prediction_Data.csv') # Read the dataset
    tumor_dataset = tumor_dataset.dropna() # Drop the rows that has empty values if found
    # (0:malignant, 1:benign)
    tumor_dataset.replace({"diagnosis": {'M': 0, 'B': 1}}, inplace=True) # Replace the M or B found in dataset with 0 and 1
    X = tumor_dataset.drop(columns=['Index', 'diagnosis'],axis=1) # Drop columns index and diagnosis from X as it will be our 30 Feature
    Y = tumor_dataset['diagnosis'] # Y will contain only the value that Models have to train/test them
    return X, Y

def splitdataset(X, Y):
    # This Function for spliting X and Y to train and test 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)  # random_state=109
    return X_train, X_test, Y_train, Y_test

def LogisticRegressionModel(x_train, x_test, y_train, y_test):
    # This Function for Logistic Regression Model
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0) # Create the Model
    model.fit(x_train, y_train) # train the Model
    # Evaluate the model
    y_pred = model.predict(x_test) # predic test data
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100) # Calculate the Accuracy
    print("Precision:", metrics.precision_score(y_test, y_pred) * 100) # Calculate the Precision
    print("Recall:", metrics.recall_score(y_test, y_pred) * 100) # Calculate the Recall
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n') # Show classification Report

def SVMModel(x_train, x_test, y_train, y_test):
    # This Function for SVM Model
    model = svm.SVC(kernel='linear') # Create the Model
    model.fit(x_train, y_train) # train the Model
    y_pred = model.predict(x_test) # predic test data
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100) # Calculate the Accuracy
    print("Precision:", metrics.precision_score(y_test, y_pred) * 100) # Calculate the Precision
    print("Recall:", metrics.recall_score(y_test, y_pred) * 100) # Calculate the Recall
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n') # Show classification Report

def GaussianNBModel(x_train, x_test, y_train, y_test):
    # This Function for Gaussian Naive Bayes Model 
    model = GaussianNB() # Create the Model
    model.fit(x_train, y_train) # train the Model
    y_pred = model.predict(x_test) # predic test data
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100) # Calculate the Accuracy
    print("Precision:", metrics.precision_score(y_test, y_pred) * 100) # Calculate the Precision
    print("Recall:", metrics.recall_score(y_test, y_pred) * 100) # Calculate the Recall
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n') # Show classification Report

def DecisionTreeModel(x_train, x_test, y_train, y_test):
    # This Function for Decision Tree Model
    model = DecisionTreeClassifier() # Create the Model
    model.fit(x_train, y_train) # train the Model
    y_pred = model.predict(x_test) # predic test data
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred) * 100) # Calculate the Accuracy
    print("Precision:", metrics.precision_score(y_test, y_pred) * 100) # Calculate the Precision
    print("Recall:", metrics.recall_score(y_test, y_pred) * 100) # Calculate the Recall
    report = classification_report(y_test, y_pred)
    print('report:', report, sep='\n') # Show classification Report

if __name__ == '__main__':
    X, Y = DatasetPreprocess()
    x_train, x_test, y_train, y_test = splitdataset(X, Y)
    print("----------------------------------------------------------------------------")
    print("|                    Welcome To Tumor Prediction System                    |")
    print("----------------------------------------------------------------------------")

    print("Please Select one of the next options:")
    print("1. Logistic Regression Model                 2. SVM Model")
    print("3. Gaussian Naive Bayes Model                4. Decision Tree Model")
    while True:
        choice = int(input("Enter Your choice:"))
        if choice == 1:
            LogisticRegressionModel(x_train, x_test, y_train, y_test)
            break
        elif choice==2:
            SVMModel(x_train, x_test, y_train, y_test)
            break
        elif choice == 3:
            GaussianNBModel(x_train, x_test, y_train, y_test)
            break
        elif choice == 4:
            DecisionTreeModel(x_train, x_test, y_train, y_test)
            break
        else:
            print("Wrong Choice!!!")
            continue