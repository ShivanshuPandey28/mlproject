import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

# Function to save a Python object to a file using pickle
def save_object(file_path, obj):
    try:
        # Create the directory if it does not exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using pickle
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
# Function to evaluate multiple models and return their performance scores
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get the model
            para = param[list(models.keys())[i]]  # Get the corresponding hyperparameters

            # Perform Grid Search to find the best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Set the best parameters and train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R-squared scores for model performance
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e, sys)
    
# Function to load a saved Python object from a file
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)  # Load and return the saved object
    except Exception as e:
        raise CustomException(e, sys)
