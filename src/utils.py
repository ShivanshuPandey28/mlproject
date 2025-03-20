import os
import sys
import dill # Importing dill for object serialization

import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_object(file_path, obj): # Function to save an object to a file
    try:
        dir_path = os.path.dirname(file_path) # Get the directory path
        
        os.makedirs(dir_path, exist_ok=True) # Create the directory if it doesn't exist
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]

                # Fit the model
                model.fit(X_train, y_train)
                
                # Predict on test data
                y_train_pred = model.predict(X_train)
                
                y_test_pred = model.predict(X_test)

                train_model_score = r2_score(y_train, y_train_pred)
                
                test_model_score = r2_score(y_test, y_test_pred)
                
                # Store the score in the report dictionary
                report[list(models.keys())[i]] = test_model_score
                
            return report

    except Exception as e:
        raise CustomException(e, sys)