# End To End MLProject
# Machine Learning Project: Student Performance Prediction

## Overview
This project is designed to predict student performance based on various input features such as gender, race/ethnicity, parental education level, lunch type, test preparation course completion, reading score, and writing score. It leverages machine learning techniques with a robust structure that includes custom exception handling, logging, and utility functions for model training and prediction pipelines. The project is built to be modular, scalable, and user-friendly, with a focus on error management and data processing efficiency.

## Features
- **Custom Exception Handling**: The `CustomException` class in `exception.py` captures and logs detailed error messages, including the script name and line number where the error occurred, enhancing debugging capabilities.
- **Logging**: The `logger.py` module configures logging to a timestamped log file in the 'logs' directory, providing detailed event tracking with timestamps, line numbers, and log levels for monitoring and troubleshooting.
- **Data Processing Utilities**: The `utils.py` file includes functions to save and load Python objects (e.g., models, preprocessors) using pickle, and an `evaluate_models` function to assess multiple models using GridSearchCV and R-squared scores, facilitating model selection and optimization.
- **Prediction Pipeline**: The `predict_pipeline.py` file contains the `PredictPipeline` class to load a trained model and preprocessor from the 'artifacts' directory and make predictions on input data, alongside a `CustomData` class to convert user input into a pandas DataFrame for seamless integration.
- **Modular Structure**: The project uses `__init__.py` files to define Python packages, ensuring maintainability, with `train_pipeline.py` as a placeholder for the training pipeline (to be implemented).

## Project Structure
- `exception.py`: Defines the `CustomException` class for error handling with detailed error messaging, including file name and line number.
- `utils.py`: Implements functions for saving/loading objects (`save_object`, `load_object`) and evaluating models (`evaluate_models`) with GridSearchCV.
- `logger.py`: Sets up logging with a dynamic log file path based on the current date and time (e.g., `2025_07_13_10_39_00.log`).
- `predict_pipeline.py`: Includes `PredictPipeline` for predictions and `CustomData` for input data framing, with paths to 'artifacts' for model and preprocessor loading.
- `__init__.py`: Marks directories as Python packages (empty files).
- `train_pipeline.py`: Reserved for the model training pipeline (currently empty, to be implemented).

## Artifacts
- artifacts/model.pkl: Saved trained machine learning model.
- artifacts/proprocessor.pkl: Saved preprocessor for data scaling and transformation.


# End-to-End Workflow

## Data Collection --> Data Preprocessing --> Model Training --> Model Prediction --> Logging and Error Handling --> Evaluation and Iteration

## 1. Data Collection
Gather student performance data including features: gender, race/ethnicity, parental level of education, lunch type, test preparation course, reading score, and writing score.
Store data in a suitable format (e.g., CSV file) for processing.
## 2. Data Preprocessing
Load the dataset using pandas.
Handle missing values, encode categorical variables (e.g., gender, race_ethnicity), and scale numerical features (e.g., reading_score, writing_score).
Save the preprocessor object using utils.save_object to artifacts/proprocessor.pkl.
## 3. Model Training
Split the preprocessed data into training and testing sets.
Define multiple machine learning models (e.g., Linear Regression, Random Forest) and their hyperparameters in train_pipeline.py.
Use utils.evaluate_models to train models with GridSearchCV, select the best model based on R-squared scores, and train it on the full training set.
Save the trained model using utils.save_object to artifacts/model.pkl.
## 4. Model Prediction
Create an instance of CustomData with new student data.
Convert the data to a DataFrame using get_data_as_data_frame.
Initialize PredictPipeline and use predict to load the model and preprocessor from 'artifacts' and generate predictions.
## 5. Logging and Error Handling
Use the logging module to record all significant steps (e.g., data loading, model training, prediction) in a timestamped log file.
Implement CustomException to catch and log errors with detailed messages during execution.
## 6. Evaluation and Iteration
Evaluate model performance using R-squared scores from utils.evaluate_models.
Iterate by adjusting hyperparameters or trying different models based on evaluation results, updating train_pipeline.py as needed.