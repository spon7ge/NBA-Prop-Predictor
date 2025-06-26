# from dotenv import load_dotenv
# import os

# load_dotenv()

class Supplier():
    def __init__(self):
        self.api_key = "3b9be537d24e5c74795a41ce44314afc"
        self.directory = '/Users/alexg/Downloads/projections.json'
    
    def getKey(self):
        return self.api_key
    
    def getDirectory(self):
        return self.directory
    
    
