import time

def readable_timestamp():
    """Generate a readable timestamp for filenames"""
    return time.strftime("%a_%b_%d_%H_%M_%S_%Y")