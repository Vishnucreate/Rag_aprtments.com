import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, request, render_template, redirect, url_for

# Load the CSV data
csv_file = 'properties.csv'  # Make sure this CSV file is in the same directory

df = pd.read_csv(csv_file)

# Preprocess the data
df['description'] = df.apply(lambda row: f"{row['RegionName']}, {row['StateName']}: Property values from {row.index[5]} to {row.index[-1]}. Latest value: {row[-1]}.", axis=1)

# Convert the dataframe to a list of dictionaries
properties = df.to_dict(orient='records')

# Load pre-trained Sentence-BERT model
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create embeddings for the property descriptions
descriptions = df['description'].tolist()
embeddings = sentence_model.encode(descriptions)

# Initialize FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Function to retrieve properties
def retrieve_properties(query, top_k=5):
    query_embedding = sentence_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [properties[i] for i in indices[0]]

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

def optimize_query(query):
    inputs = tokenizer.encode("Optimize this search query: " + query, return_tensors='pt')
    outputs = gpt_model.generate(inputs, max_length=50, num_return_sequences=1)
    optimized_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return optimized_query

def generate_detailed_response(query):
    optimized_query = optimize_query(query)
    retrieved_docs = retrieve_properties(optimized_query)
    
    # Prepare input for the generative model
    context = " ".join([doc['description'] for doc in retrieved_docs])
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate response
    max_length = inputs.shape[1] + 100
    outputs = gpt_model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, retrieved_docs

# Flask application
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    response, retrieved_docs = generate_detailed_response(query)
    
    # For simplicity, let's assume each property has a URL field or display information
    results_html = '<ul>' + ''.join([f'<li>{doc["RegionName"]}, {doc["StateName"]} - Latest Value: {doc["2024-06-30"]}</li>' for doc in retrieved_docs]) + '</ul>'
    return render_template('results.html', response=response, results_html=results_html)

if __name__ == '__main__':
    app.run(debug=True)
