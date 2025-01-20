from pymongo import MongoClient
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Initialize the MongoDB client
client = MongoClient('localhost', 27017)
db = client['fifa']
collection = db['players']

# Load the data from CSV
def load_data():
    data = pd.read_csv('data/player_data.csv')
    data = data.drop('url', axis=1, inplace=True)
    return data

# Save the embeddings to MongoDB
def process_and_store(dataframe):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    for idx, row in dataframe.iterrows():
        text = row['content']
        embedding = model.encode(text).tolist()
        doc = {
            "id": int(row['id']) if 'id' in row else idx,
            "title": row.get('title', f"Document {idx}"),
            "text": text,
            "embedding": embedding
        }
        collection.insert_one(doc)