# from dotenv import load_dotenv
# import os

# load_dotenv()

class Supplier():
    def __init__(self):
        self.api_key = '2a1f2c073b529f516df87b0e8a459946'
        self.directory = '/Users/alexg/Downloads/projections.json'
    
    def getKey(self):
        return self.api_key
    
    def getDirectory(self):
        return self.directory
    
    
