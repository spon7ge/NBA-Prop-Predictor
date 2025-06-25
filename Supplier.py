# from dotenv import load_dotenv
# import os

# load_dotenv()

class Supplier():
    def __init__(self):
        self.api_key = "a291ae2e07e29b8ce3d0793cbb340d07"
        self.directory = '/Users/alexg/Downloads/projections.json'
    
    def getKey(self):
        return self.api_key
    
    def getDirectory(self):
        return self.directory
    
    
