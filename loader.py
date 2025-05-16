import os
import json
from pathlib import Path
import textwrap


ARTICLES_DIR = "articles"

OUTPUT_PATH = "data/chunks.json"


Path("data").mkdir(parents=True, exist_ok=True)

def split_text(text, max_words=200):
    """Split text into chunks of ~200 words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    return chunks

def process_articles():
    all_chunks = []

    for filename in os.listdir(ARTICLES_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(ARTICLES_DIR, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                full_text = file.read()

            source = filename.replace(".txt", "").replace("_", " ").title()
            chunks = split_text(full_text)

            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{source}_{idx}",
                    "source": source,
                    "text": chunk
                })

    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        json.dump(all_chunks, out_file, indent=4)

    print(f" Processed {len(all_chunks)} chunks from {len(os.listdir(ARTICLES_DIR))} articles.")
    print(f" Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_articles()
