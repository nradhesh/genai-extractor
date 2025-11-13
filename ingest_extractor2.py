import os
import glob
import hashlib
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = "vector_db_v2"
SOURCE_DIR = "extractor-2"

def normalise_faculty_name(raw_name: str) -> str:
    name = raw_name or ""
    name = name.replace("google_scholar_", "")
    name = name.replace("_", " ")
    name = name.replace("  ", " ").strip()
    if name.lower().startswith("dr "):
        name = name.replace("Dr ", "Dr. ", 1)
    return name.strip()

def prepare_fields(row: dict) -> dict:
    title = str(row.get("title", "")).strip()
    authors = str(row.get("authors", "")).strip()
    publication = str(row.get("publication", "")).strip()
    year_val = str(row.get("year", "")).strip()
    year_int = None
    try:
        year_int = int(year_val[:4])
    except Exception:
        year_int = None
    cited_by = row.get("cited_by.value")
    citations_int = None
    if pd.notna(cited_by):
        try:
            citations_int = int(float(cited_by))
        except Exception:
            try:
                citations_int = int(str(cited_by))
            except Exception:
                citations_int = None
    citations_display = str(citations_int) if citations_int is not None else "N/A"
    return {
        "title": title,
        "authors": authors,
        "publication": publication,
        "year": year_val,
        "year_int": year_int,
        "citations_int": citations_int,
        "citations_display": citations_display,
        "link": str(row.get("link", "")),
        "citation_id": str(row.get("citation_id", "")),
    }

def build_text(faculty: str, pretty_faculty: str, fields: dict) -> str:
    parts = [
        f"Faculty: {pretty_faculty}" if pretty_faculty else None,
        f"FacultyID: {faculty}" if faculty else None,
        f"Title: {fields['title']}" if fields.get("title") else None,
        f"Authors: {fields['authors']}" if fields.get("authors") else None,
        f"Publication: {fields['publication']}" if fields.get("publication") else None,
        f"Year: {fields['year']}" if fields.get("year") else None,
        f"Citations: {fields['citations_display']}",
    ]
    return "\n".join([p for p in parts if p])

def build_id(row: dict, faculty: str) -> str:
    base = f"{faculty}|{row.get('citation_id','')}|{row.get('title','')}".encode("utf-8", errors="ignore")
    return hashlib.sha1(base).hexdigest()

def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)

    csv_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.csv")))
    if not csv_paths:
        print(f"No CSV files found in {SOURCE_DIR}")
        return

    total_added = 0
    for path in csv_paths:
        faculty = os.path.splitext(os.path.basename(path))[0]
        pretty_faculty = normalise_faculty_name(faculty)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue

        texts = []
        metadatas = []
        ids = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            fields = prepare_fields(row_dict)
            text = build_text(faculty, pretty_faculty, fields)
            if not text.strip():
                continue
            rec_id = build_id(row_dict, faculty)
            meta = {
                "faculty": faculty,
                "faculty_pretty": pretty_faculty,
                "title": fields["title"],
                "authors": fields["authors"],
                "publication": fields["publication"],
                "year": fields["year"],
                "year_int": fields["year_int"],
                "citations": fields["citations_int"],
                "link": fields["link"],
                "citation_id": fields["citation_id"],
            }
            texts.append(text)
            metadatas.append(meta)
            ids.append(rec_id)

        if texts:
            # Add in batches to avoid duplicates by IDs (Chroma ignores identical IDs)
            vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            total_added += len(texts)
            print(f"Added {len(texts)} records from {faculty}")

    # Chroma in langchain_chroma persists automatically when configured with persist_directory
    print(f"Done. Added ~{total_added} records into '{DB_DIR}'.")

if __name__ == "__main__":
    main()


