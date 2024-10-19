import os
from dotenv import load_dotenv


load_dotenv()


LOGGING_LEVEL = os.getenv('LOGGING_LEVEL') or 'WARNING'
