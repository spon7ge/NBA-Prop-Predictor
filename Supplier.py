from dotenv import load_dotenv
import os

load_dotenv()

class Supplier():
    def __init__(self):
        self.api_key = os.environ.get('API_KEY')
        self.directory = os.environ.get('PROJECTIONS_DIRECTORY')
    
    def getKey(self):
        return self.api_key
    
    def getDirectory(self):
        return self.directory
    
    
