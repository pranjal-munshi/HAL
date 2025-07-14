from flask import Flask, request, jsonify, render_template, redirect
from retriever.query_module import RetrieveHelicopterManualInfo
from indexing.faiss_manager import FAISSRetriever
from sentence_transformers import SentenceTransformer
import faiss
import json
import os

app = Flask(__name__)

# =================== Load Model, Index, Metadata ===================

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index_path = "data/index.faiss"
index = faiss.read_index(index_path)

# Load metadata
with open("data/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

retriever = FAISSRetriever(index, model, metadata, k=3)
query_module = RetrieveHelicopterManualInfo(retriever)

# =================== Routes ===================

@app.route("/")
def home():
    return redirect("/login")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        if username == "admin" and password == "admin":
            return redirect("/frontend")
        else:
            return render_template("HAL-login.html", error="Invalid credentials")
    return render_template("HAL-login.html")

@app.route("/frontend")
def frontend():
    return render_template("HAL-frontend.html")

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Please provide a 'question' key in JSON."}), 400
    question = data["question"]
    result = query_module(question)
    return jsonify(result)

# =================== Run ===================

if __name__ == "__main__":
    app.run(debug=True)
