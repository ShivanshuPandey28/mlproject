import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs")  # Directory for logs
os.makedirs(logs_path, exist_ok=True)  # Create the directory if it does not exist

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)  # Full path for the log file

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] - %(lineno)d - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# if __name__ == "__main__": #To test the exception
#     logging.info("Logging started")