import os
import sys
import dill # Importing dill for object serialization

import numpy as np
import pandas as pd 

from src.exception import CustomException

def save_object(file_path, obj): # Function to save an object to a file
    try:
        dir_path = os.path.dirname(file_path) # Get the directory path
        
        os.makedirs(dir_path, exist_ok=True) # Create the directory if it doesn't exist
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)