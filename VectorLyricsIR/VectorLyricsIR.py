import os
import re
import math
from collections import defaultdict, Counter

# Global data structures
term_doc_freq = defaultdict(Counter)  # term -> doc_id -> term frequency
doc_map = {}  # doc_id -> "artist/song"
doc_lengths = {}  # doc_id -> precomputed vector length for normalization
doc_id_counter = 0

stopwords = set([
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "in", "on", "at", "for", "with", "of", "to", "by", "as", "from"
])

def preprocess(text):
    """
    Converts text to lowercase, removes non-alphanumeric characters,
    and splits into a list of words (tokens).
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return [token for token in tokens if token not in stopwords]

def build_index(base_path):
    """
    Builds the TF index and computes the document vector lengths for all lyrics files.
    Scans each artist directory and their corresponding .txt files.
    """
    global doc_id_counter
    for artist in os.listdir(base_path):
        artist_path = os.path.join(base_path, artist)
        if not os.path.isdir(artist_path):
            continue
        for filename in os.listdir(artist_path):
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(artist_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            doc_id_counter += 1
            doc_id = doc_id_counter
            song_title = os.path.splitext(filename)[0]
            doc_map[doc_id] = f"{artist}/{song_title}"

            tokens = preprocess(content)
            tf = Counter(tokens)
            for term, freq in tf.items():
                term_doc_freq[term][doc_id] = freq

    # Precompute vector length (L2 norm) for each document
    for doc_id in doc_map:
        length = 0.0
        for term in term_doc_freq:
            tf = term_doc_freq[term][doc_id]
            if tf == 0:
                continue
            df = len(term_doc_freq[term])
            idf = math.log((1 + len(doc_map)) / (1 + df)) + 1
            length += (tf * idf) ** 2
        doc_lengths[doc_id] = math.sqrt(length)

def compute_query_vector(query_terms):
    """
    Constructs the query vector using TF-IDF weights.
    Returns a dictionary: term -> weighted tf-idf value.
    """
    tf = Counter(query_terms)
    query_vec = {}
    for term, freq in tf.items():
        df = len(term_doc_freq[term]) if term in term_doc_freq else 0
        idf = math.log((1 + len(doc_map)) / (1 + df)) + 1
        query_vec[term] = freq * idf
    return query_vec

def cosine_similarity(query_vec):
    """
    Computes cosine similarity between the query vector and all documents.
    Returns a list of (doc_id, score) tuples sorted in descending order of score.
    """
    scores = defaultdict(float)
    query_norm = math.sqrt(sum(weight ** 2 for weight in query_vec.values()))  # Normalize query vector

    for term, q_weight in query_vec.items():
        if term not in term_doc_freq:
            continue
        for doc_id, tf in term_doc_freq[term].items():
            df = len(term_doc_freq[term])
            idf = math.log((1 + len(doc_map)) / (1 + df)) + 1
            scores[doc_id] += q_weight * (tf * idf)  # Dot product

    # Final cosine normalization using document length and query length
    for doc_id in scores:
        if doc_lengths[doc_id] != 0 and query_norm != 0:
            scores[doc_id] /= (doc_lengths[doc_id] * query_norm)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def print_results(ranked_docs):
    """
    Displays the top-ranked documents with their scores.
    If no results are found, prints a message accordingly.
    """
    if not ranked_docs:
        print("No results found.")
    else:
        for doc_id, score in ranked_docs[:10]:
            print(f"{doc_map[doc_id]} (score: {score:.4f})")

def main():
    """
    Entry point of the program.
    Builds index, takes user input, performs vector-based search, and prints results.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "Lyrics")
    base_path = os.path.abspath(base_path)
    print(f"Indexing path: {base_path}")
    build_index(base_path)
    print("LyricsIR Vector Space Search System")
    while True:
        query = input("Enter query (or 'exit'): ")
        if query == "exit":
            break
        query_terms = preprocess(query)
        query_vec = compute_query_vector(query_terms)
        ranked = cosine_similarity(query_vec)
        print_results(ranked)

if __name__ == "__main__":
    main()
