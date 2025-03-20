import os
import sys
from src.exception import CustomException  # Custom exception handling module
from src.logger import logging  # Logging module for tracking execution
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation  # Importing Data Transformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig  # Importing Model Trainer
from src.components.model_trainer import ModelTrainer

# Configuration class for defining file paths for data storage
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to store training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to store testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path to store raw data

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()  # Initialize ingestion config

    def initiate_data_ingestion(self):
        """
        This function reads the dataset, splits it into training and testing sets,
        and saves them as CSV files.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read dataset into a pandas DataFrame
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            # Create directory for saving data if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Split data into training (80%) and testing (20%)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save training and testing datasets as CSV files
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)  # Handle exceptions

# Running the data ingestion pipeline
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Initiate data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Initiate model training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))
