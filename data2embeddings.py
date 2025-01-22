from pymongo import MongoClient
import numpy as np
import pandas as pd
import os
from sentence_transformers import SentenceTransformer

# Initialize the MongoDB client
client = MongoClient('localhost', 27017)
db = client['fifa_personal_project']
row_collection = db['row_embeddings']
attribute_collection = db['attribute_embeddings']

# Load the data from CSV
def load_data():
    data = pd.read_csv('player_data.csv')
    data.drop('url', axis=1, inplace=True)
    data.drop('Unnamed: 0.1', axis=1, inplace=True)
    data.drop('Unnamed: 0', axis=1, inplace=True)

    row_columns = ['Name', 'Rank', 'Position', 'Team', 'Age', 'Nation', 'League',
                   'PAC', 'SHO', 'PAS', 'DRI', 'DEF', 'PHY', 'Acceleration', 'Sprint Speed',
                         'Positioning', 'Finishing', 'Shot Power', 'Long Shots', 'Volleys', 'Penalties']

    return data, row_columns

# Save the embeddings to MongoDB
def process_and_store(dataframe, row_columns):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    
    for idx, row in dataframe.iterrows():

        # generate row embeddings here
        row_text = ' '.join([str(row[col]) for col in row_columns])
        row_embedding = model.encode(row_text).tolist()

        row_doc = {
            "id": idx,
            "Name" : row['Name'],
            "Rank" : row['Rank'],
            "Position" : row['Position'],
            "Team" : row["Team"],
            "Age" : row["Age"],
            "Nation" : row["Nation"],
            "League" : row["League"],
            "row_embedding" : row_embedding,
            "PAC": row['PAC'],
            "SHO": row['SHO'],
            "PAS": row['PAS'],
            "DRI": row['DRI'],
            "DEF": row['DEF'],
            "PHY": row['PHY'],
            "Acceleration": row['Acceleration'],
            "Sprint Speed": row['Sprint Speed'],
            "Positioning": row['Positioning'],
            "Finishing": row['Finishing'],
            "Shot Power": row['Shot Power'],
            "Long Shots": row['Long Shots'],
            "Volleys": row['Volleys'],
            "Penalties": row['Penalties'],
        }


        try:
            row_collection.insert_one(row_doc)
        except Exception as e:
            print(f"Error inserting row_doc at index {idx}: {e}")




if __name__ == '__main__':
    data, row_columns = load_data()
    process_and_store(data, row_columns)