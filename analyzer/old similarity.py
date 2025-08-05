from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

def compare_responses(file1, file2, out_path):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    vectorizer = TfidfVectorizer()
    results = []

    for i, (resp1, resp2) in enumerate(zip(lines1, lines2)):
        vecs = vectorizer.fit_transform([resp1.strip(), resp2.strip()])
        score = cosine_similarity(vecs[0], vecs[1])[0][0]
        results.append((i+1, score))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Example", "CosineSimilarity"])
        writer.writerows(results)

    print(f"Saved results to {out_path}")
