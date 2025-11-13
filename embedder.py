# create_embeddings.py
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "google_scholar_RCA_Naidu.csv"
DB_DIR = "vector_db"

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(CSV_FILE)

# Create text chunks for embedding
texts = [
    f"Title: {row['title']}\nAuthors: {row['authors']}\nPublication: {row.get('publication','')}\nYear: {row['year']}\nCitations: {row.get('cited_by_value','N/A')}"
    for _, row in df.iterrows()
]

# -----------------------------
# CREATE EMBEDDINGS AND VECTOR DB
# -----------------------------
print("🔹 Creating embeddings...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Build and persist the vector DB
vectordb = Chroma.from_texts(texts, embedding=embedding_model, persist_directory=DB_DIR)
vectordb.persist()

print(f"✅ Embeddings stored successfully in '{DB_DIR}' directory.")
