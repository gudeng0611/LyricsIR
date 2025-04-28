import os
import re
import math
import numpy as np
from collections import defaultdict, Counter

# Global data structures
term_doc_freq = defaultdict(Counter)
doc_map = {}         # doc_id -> "Artist/SongTitle"
doc_lengths = {}     # doc_id -> vector length
doc_id_counter = 0

lsi_vocab = []
term_index_map = {}
doc_index_map = {}
U_k = None
s_k = None
Vt_k = None
doc_latent = None
K_DIM = 100  # latent dimension

stopwords = set([
    "a","an","the","is","are","was","were","am","be","been","being",
    "and","or","but","if","then","this","that","these","those",
    "in","on","at","for","with","of","to","by","as","from"
])

# Relevant set definitions
# use "Artist/SongTitle" strings for relevance
relevant_docs_str = {
    'happy': {
        "Bruno Mars/Just The Way You Are",
        "Bruno Mars/Count On Me",
        "Bruno Mars/Marry You",
        "Bruno Mars/The Lazy Song",
        "Coldplay/A Sky Full Of Stars",
        "Coldplay/Adventure Of A Lifetime",
        "Coldplay/Yellow",
        "Coldplay/Paradise",
        "Ed Sheeran/Perfect",
        "Ed Sheeran/Shape Of You",
        "Ed Sheeran/Shivers",
        "Justin Bieber/What Do You Mean",
        "Katy Perry/Firework",
        "Katy Perry/Roar",
        "Katy Perry/Teenage Dream",
        "Kendrick Lamar/Alright",
        "Kendrick Lamar/Humble",
        "Kendrick Lamar/King Kunta",
        "Kendrick Lamar/N95",
        "Lady Gaga/Shallow",
        "Lady Gaga/Poker Face",
        "Lady Gaga/Telephone",
        "Mariah Carey/All I Want For Christmas Is You",
        "Mariah Carey/Vision Of Love",
        "Maroon 5/Sugar",
        "Maroon 5/Animals",
        "Maroon 5/Girls Like You",
        "Maroon 5/Sunday Morning",
        "Michael Jackson/Heal The World",
        "Michael Jackson/Man In The Mirror",
        "Michael Jackson/We Are The World",
        "Queen/Don't Stop Me Now",
        "Queen/We Are The Champions",
        "Queen/We Will Rock You",
        "Redbone/Come And Get Your Love",
        "Taylor Swift/Shake It Off",
        "Taylor Swift/Blank Space",
        "The Beatles/I Want To Hold Your Hand",
        "The Beatles/Let It Be",
        "The Beatles/Come Together",
        "The Weekend/Blinding Lights",
        "The Weekend/Starboy",
        "Travis Scott/Goosebumps",
        "Travis Scott/Butterfly Effect",
        "Travis Scott/Antidote",
    },
    'sad': {
        "Adele/Easy On Me",
        "Adele/Hello",
        "Adele/Rolling In The Deep",
        "Adele/Someone Like You",
        "Billie Eilish/When The Party's Over",
        "Billie Eilish/Bury A Friend",
        "Bruno Mars/When I Was Your Man",
        "Coldplay/Fix You",
        "Coldplay/The Scientist",
        "Ed Sheeran/Photograph",
        "Eminem/Love The Way You Lie",
        "Justin Bieber/Sorry",
        "Justin Bieber/Love Yourself",
        "Justin Bieber/Stay",
        "Lady Gaga/Million Reasons",
        "Mariah Carey/We Belong Together",
        "Maroon 5/Won't Go Home Without You",
        "Queen/Bohemian Rhapsody",
        "Taylor Swift/Bad Blood",
        "The Beatles/Yesterday",
        "The Weekend/Save Your Tears",
    }
}

# Utility functions
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in stopwords]

def precision_at_k(ranked, relevant_set, k=5):
    topk = [doc_id for doc_id, _ in ranked[:k]]
    if not relevant_set:
        return 0.0
    return sum(1 for d in topk if d in relevant_set) / k

def average_precision(ranked, relevant_set):
    if not relevant_set:
        return 0.0
    hits = 0
    total = 0.0
    for i, (doc_id, _) in enumerate(ranked, start=1):
        if doc_id in relevant_set:
            hits += 1
            total += hits / i
    return total / len(relevant_set)

def print_results(label, ranked):
    print(f"\n{label} Results:")
    for doc_id, score in ranked[:10]:
        print(f"  {doc_map[doc_id]} (score: {score:.4f})")

# Indexing & VSM Model
def build_index(base_path):
    global doc_id_counter
    for artist in os.listdir(base_path):
        artist_path = os.path.join(base_path, artist)
        if not os.path.isdir(artist_path):
            continue
        for fname in os.listdir(artist_path):
            if not fname.endswith(".txt"):
                continue
            full = os.path.join(artist_path, fname)
            with open(full, "r", encoding="utf-8") as f:
                content = f.read()
            doc_id_counter += 1
            doc_map[doc_id_counter] = f"{artist}/{os.path.splitext(fname)[0]}"
            tokens = preprocess(content)
            tf = Counter(tokens)
            for term, freq in tf.items():
                term_doc_freq[term][doc_id_counter] = freq
    # compute document lengths
    N = len(doc_map)
    for d in doc_map:
        s = 0.0
        for term, freqs in term_doc_freq.items():
            tfv = freqs.get(d, 0)
            if tfv == 0:
                continue
            df = len(freqs)
            idf = math.log((1+N)/(1+df)) + 1
            s += (tfv * idf) ** 2
        doc_lengths[d] = math.sqrt(s)

def vsm_search(query_terms):
    N = len(doc_map)
    qvec = {}
    tfq = Counter(query_terms)
    for t, f in tfq.items():
        df = len(term_doc_freq.get(t, {}))
        idf = math.log((1+N)/(1+df)) + 1
        qvec[t] = f * idf
    scores = defaultdict(float)
    qnorm = math.sqrt(sum(w*w for w in qvec.values()))
    for t, w in qvec.items():
        for d, tfv in term_doc_freq.get(t, {}).items():
            df = len(term_doc_freq[t])
            idf = math.log((1+N)/(1+df)) + 1
            scores[d] += w * (tfv * idf)
    for d in scores:
        if qnorm and doc_lengths[d]:
            scores[d] /= (qnorm * doc_lengths[d])
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# LSI Model
def build_lsi_model():
    global lsi_vocab, term_index_map, doc_index_map, U_k, s_k, Vt_k, doc_latent
    lsi_vocab = list(term_doc_freq.keys())
    term_index_map = {t:i for i,t in enumerate(lsi_vocab)}
    doc_ids = list(doc_map.keys())
    doc_index_map = {d:i for i,d in enumerate(doc_ids)}

    M = np.zeros((len(lsi_vocab), len(doc_ids)))
    N = len(doc_ids)
    for term, freqs in term_doc_freq.items():
        i = term_index_map[term]
        df = len(freqs)
        idf = math.log((1+N)/(1+df)) + 1
        for d, tfv in freqs.items():
            j = doc_index_map[d]
            M[i,j] = tfv * idf

    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(K_DIM, len(s))
    U_k = U[:,:k]
    s_k = s[:k]
    Vt_k = Vt[:k,:]
    doc_latent = np.diag(s_k) @ Vt_k

def lsi_search(query_terms):
    N = len(doc_map)
    qvec = np.zeros(len(lsi_vocab))
    tfq = Counter(query_terms)
    for t, f in tfq.items():
        if t in term_index_map:
            i = term_index_map[t]
            df = len(term_doc_freq[t])
            idf = math.log((1+N)/(1+df)) + 1
            qvec[i] = f * idf

    q_latent = (U_k.T @ qvec) / s_k
    qnorm = np.linalg.norm(q_latent)
    scores = {}
    for d, j in doc_index_map.items():
        dvec = doc_latent[:, j]
        dn = np.linalg.norm(dvec)
        scores[d] = (np.dot(q_latent, dvec) / (qnorm * dn)) if qnorm and dn else 0.0
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Main
def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Lyrics"))
    print("Indexing:", base)
    build_index(base)

    # invert doc_map for name->id lookup
    name_to_id = {name: did for did, name in doc_map.items()}

    # convert string-based relevances to id-based
    relevant_docs = {}
    for q, names in relevant_docs_str.items():
        relevant_docs[q] = { name_to_id[n] for n in names if n in name_to_id }

    build_lsi_model()

    while True:
        query = input("\nEnter query (or 'exit'): ").strip().lower()
        if query == "exit":
            break
        terms = preprocess(query)

        vsm_res = vsm_search(terms)
        lsi_res = lsi_search(terms)

        print_results("VSM", vsm_res)
        print_results("LSI", lsi_res)

        # evaluate if relevant-set exists
        rel = relevant_docs.get(query, set())
        if rel:
            p_vsm  = precision_at_k(vsm_res, rel)
            ap_vsm = average_precision(vsm_res, rel)
            p_lsi  = precision_at_k(lsi_res, rel)
            ap_lsi = average_precision(lsi_res, rel)
            print(f"\nMetrics (P@5, MAP):")
            print(f"  VSM -> P@5: {p_vsm:.3f}, MAP: {ap_vsm:.3f}")
            print(f"  LSI -> P@5: {p_lsi:.3f}, MAP: {ap_lsi:.3f}")
        else:
            print("No prepared relevant-set for this query.")


if __name__ == "__main__":
    main()
