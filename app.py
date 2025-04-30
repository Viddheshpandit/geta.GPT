# === Step 1: Load Dataset from CSV ===
import csv
import json
from datetime import datetime
import os

shlok_dataset = []
with open("C:/code/python/divine/BHAGAVAD_GITA_DATASET.csv", encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        shlok_dataset.append({
            "shlok": row.get("SANSKRIT VERSE", ""),
            "chapter": int(row.get("CHAPTER", 0)),
            "verse": int(row.get("VERSE NUMBER", 0)),
            "book": "Bhagavad Gita",
            "translation": row.get("VERSE TRANSLATION", ""),
            "tags": [],  # Tags not provided in dataset
            "explanation": row.get("VERSE COMMENTARY", "")
        })

# === Step 2: Semantic Search Function Using Sentence Transformers ===
from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare the database embeddings
corpus = [entry["explanation"] + " " + entry["translation"] for entry in shlok_dataset]
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

def find_matching_shlok(user_query, threshold=0.30):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, corpus_embeddings)[0].cpu().numpy()
    best_match_idx = int(np.argmax(similarities))
    best_score = similarities[best_match_idx]

    if best_score < threshold:
        return None  # Not a meaningful query
    return shlok_dataset[best_match_idx]

# === Step 3: Flask Web API ===
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# === Route for Home Page ===
@app.route('/')
def home():
    return render_template('index.html')

# === Route to Handle User's Query and Save it ===
@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required."}), 400

    result = find_matching_shlok(user_query)

    if result is None:
        return jsonify({"error": "Your question doesn't match any relevant guidance from the Bhagavad Gita. Please rephrase it."}), 404

    # === Save the query to CSV ===
    save_path = "user_queries.csv"
    file_exists = os.path.isfile(save_path)

    with open(save_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['timestamp', 'user_query', 'chapter', 'verse', 'shlok', 'translation'])

        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_query,
            result['chapter'],
            result['verse'],
            result['shlok'],
            result['translation']
        ])

    return jsonify(result)

# === Running the App ===
if __name__ == '__main__':
    app.run(debug=True)
