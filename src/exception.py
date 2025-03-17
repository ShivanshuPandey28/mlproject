import sys
import logging
from src.logger import logging  # If error is occured then log file will be created in logs folder

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename #tb_frame provides the execution frame, #f_code contains code-related information,      #co_filename contains the name of the file (script) where the error occurred.
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message
    
class CustomException(Exception):
    def __init__(self,error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail = error_detail)

    def __str__(self):
        return self.error_message
    

# if __name__ == "__main__":  #To test the exception
#         try:
#             a = 1/0
#         except Exception as e:
#             logging.info("Divide by zero error")
#             raise CustomException(e,sys)    