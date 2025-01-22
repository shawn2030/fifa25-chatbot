from flask import Flask, request, render_template
import numpy as np
import faiss
from pymongo import MongoClient
import openai
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# MongoDB setup
client = MongoClient('localhost', 27017)
db = client['fifa_personal_project']
row_collection = db['row_embeddings']


def load_faiss_index(collection_name, key_field):
    documents = list(collection_name.find({}, {key_field: 1, "id": 1}))
    embeddings = np.array([doc[key_field] for doc in documents]).astype(np.float32)
    ids = np.array([doc['id'] for doc in documents])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, ids

# Load FAISS indices
row_faiss_index, row_doc_ids = load_faiss_index(row_collection, "row_embedding")

openai.api_key = os.getenv("OPENAI_API_KEY") 

def get_llm_answer(query, contexts):
    context_text = "\n".join(contexts)
    prompt = f"Answer the following question based on the given context:\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7,
        )
        answer = response['choices'][0]['message']['content'].strip()
        return answer
    except Exception as e:
        print("Error with OpenAI API:", e)
        return "Sorry, I couldn't process your request."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form.get('query')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query_text).astype(np.float32)
    print(query_text)

    # Search row embeddings
    row_distances, row_indices = row_faiss_index.search(np.array([query_embedding]), 1)
    row_results = []
    for idx in row_indices[0]:
        document = row_collection.find_one({"id": int(row_doc_ids[idx])}, {"_id": 0, "Name": 1, "Team": 1,"Rank": 1,"Position": 1, "Team": 1, "Age": 1, "Nation": 1, "League": 1, 'PAC': 1, 'SHO': 1, 'PAS': 1, 'DRI': 1, 'DEF': 1, 'PHY': 1, 'Acceleration': 1, 'Sprint Speed': 1,
                         'Positioning': 1, 'Finishing': 1, 'Shot Power': 1, 'Long Shots': 1, 'Volleys': 1, 'Penalties': 1})
        print(document)
        if document:
            row_results.append(f"Player Name: {document['Name']}, Team: {document['Team']}, Rank: {document['Rank']}, Position: {document['Position']}, Age: {document['Age']}, Nation: {document['Nation']}, League: {document['League']} and the Attributes are : Pace: {document['PAC']}, Shooting: {document['SHO']}, Passing: {document['PAS']}, Dribble: {document['DRI']}, Defending: {document['DEF']}, Physique: {document['PHY']}, Acceleration: {document['Acceleration']}, Sprint Speed: {document['Sprint Speed']}, Positioning: {document['Positioning']}, Finishing: {document['Finishing']}, Shot Power: {document['Shot Power']}, Long Shots: {document['Long Shots']}, Volleys: {document['Volleys']}, Penalties: {document['Penalties']}")

    combined_contexts = row_results
    print(combined_contexts)
    answer = get_llm_answer(query_text, combined_contexts)

    return render_template(
        'index.html',
        query=query_text,
        row_results=row_results,
        answer=answer
    )

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=7050, debug=True)
