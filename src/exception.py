import sys

def error_message_detail(error, error_detail: sys):
    _,_, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = f"Error occured in script: [{file_name}] at line number[{exc_tb.tb_lineno}] with message: [{str(error)}]"

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self) -> str:
        return CustomException.__name__ + f"({self.error_message})"
