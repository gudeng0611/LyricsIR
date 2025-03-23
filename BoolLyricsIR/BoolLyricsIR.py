import os
import re
from collections import defaultdict

# Inverted index for boolean search: term -> doc_id -> [positions]
inverted_index = defaultdict(lambda: defaultdict(list))
# Mapping from internal doc_id to "Artist/Song Title"
doc_map = {}
doc_id_counter = 0

# List of stopwords to filter out from indexing and search
stopwords = set([
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "and", "or", "but", "if", "then", "this", "that", "these", "those",
    "in", "on", "at", "for", "with", "of", "to", "by", "as", "from"
])

def preprocess(text):
    """
    Preprocess the input text: convert to lowercase, remove non-alphanumeric characters,
    split into tokens, and remove stopwords.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return [token for token in tokens if token not in stopwords]

def build_index(base_path):
    """
    Build an inverted index from lyrics files. Each artist has a folder containing song text files.
    Each word is mapped to its positions in each document.
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
            for pos, token in enumerate(tokens):
                inverted_index[token][doc_id].append(pos)

def search_term(term):
    """
    Return a set of document IDs that contain the specified term.
    """
    return set(inverted_index[term].keys()) if term in inverted_index else set()

def set_union(a, b):
    """
    Return the union of two sets (used for OR logic).
    """
    return a.union(b)

def set_intersection(a, b):
    """
    Return the intersection of two sets (used for AND logic).
    """
    return a.intersection(b)

def set_difference(a, b):
    """
    Return the set difference (used for NOT logic).
    """
    return a.difference(b)

def boolean_search(query):
    """
    Parse and evaluate a boolean query. Supports formats like:
    - "word"
    - "not word"
    - "word1 and word2", "word1 or word2"
    """
    tokens = query.lower().split()
    all_docs = set(doc_map.keys())
    if len(tokens) == 1:
        return search_term(tokens[0])
    elif len(tokens) == 2 and tokens[0] == "not":
        return set_difference(all_docs, search_term(tokens[1]))
    elif len(tokens) == 3:
        set1 = search_term(tokens[0])
        set2 = search_term(tokens[2])
        if tokens[1] == "and":
            return set_intersection(set1, set2)
        elif tokens[1] == "or":
            return set_union(set1, set2)
    return set()

def phrase_search(phrase_raw):
    """
    Search for exact phrases appearing in documents by checking positional indexes.
    Returns set of document IDs where the phrase appears consecutively.
    """
    words = preprocess(phrase_raw)
    if not words:
        return set()
    result = set()
    first_postings = inverted_index.get(words[0], {})
    for doc_id, positions in first_postings.items():
        for pos in positions:
            match = True
            for offset, word in enumerate(words[1:], 1):
                post_map = inverted_index.get(word, {})
                if doc_id not in post_map:
                    match = False
                    break
                if (pos + offset) not in post_map[doc_id]:
                    match = False
                    break
            if match:
                result.add(doc_id)
                break
    return result

def print_results(doc_ids):
    """
    Print the search results in "Artist/Song Title" format.
    """
    if not doc_ids:
        print("No results found.")
    else:
        for doc_id in sorted(doc_ids):
            print(doc_map[doc_id])

def main():
    """
    Entry point of the Boolean IR system. Builds index and handles user queries
    with support for boolean and phrase search.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "Lyrics")
    base_path = os.path.abspath(base_path)
    print(f"Indexing path: {base_path}")
    build_index(base_path)
    print("LyricsIR Boolean Search System")
    while True:
        query = input("Enter query (or 'exit'): ").lower()
        if query == "exit":
            break
        if query.startswith('"') and query.endswith('"'):
            result = phrase_search(query[1:-1])
        else:
            result = boolean_search(query)
        print_results(result)

if __name__ == "__main__":
    main()