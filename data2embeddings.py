from pymongo import MongoClient
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Initialize the MongoDB client
client = MongoClient('localhost', 27017)
db = client['fifa']
row_collection = db['row_embeddings']
attribute_collection = db['attribute_embeddings']

# Load the data from CSV
def load_data():
    data = pd.read_csv('player_data.csv')
    data.drop('url', axis=1, inplace=True)
    data.drop('Unnamed: 0.1', axis=1, inplace=True)
    data.drop('Unnamed: 0', axis=1, inplace=True)

    row_columns = ['Name', 'Rank', 'Position', 'Team', 'Age', 'Nation', 'League']
    attribute_columns = ['PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 'Acceleration', 'Sprint Speed',
                         'Positioning', 'Finishing', 'Shot Power', 'Long Shots', 'Volleys', 'Penalties']

    return data, row_columns, attribute_columns

# Save the embeddings to MongoDB
def process_and_store(dataframe, row_columns, attribute_columns):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    for idx, row in dataframe.iterrows():

        # generate row embeddings here
        row_text = ' '.join([str(row[col]) for col in row_columns if col in row_columns])
        row_embedding = model.encode(row_text).tolist()

        attribute_text = ' '.join([str(row[col]) for col in attribute_columns if col in attribute_columns])
        attribute_embedding  = model.encode(attribute_text).tolist()

        row_doc = {
            "id": idx,
            "Name" : row['Name'],
            "Team" : row["Team"],
            "row_embedding" : row_embedding 
        }

        attribute_doc = {
            "id": idx,
            "Name": row['Name'],
            "attribute_embedding": attribute_embedding
        }

        row_collection.insert_one(row_doc)
        attribute_collection.insert_one(attribute_doc)


if __name__ == '__main__':
    data, row_columns, attribute_columns = load_data()
    process_and_store(data, row_columns, attribute_columns)