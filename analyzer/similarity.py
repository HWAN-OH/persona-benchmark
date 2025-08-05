
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import csv

def compare_responses(file1, file2, out_path):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    vectorizer = TfidfVectorizer()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    results = []

    for i, (resp1, resp2) in enumerate(zip(lines1, lines2)):
        resp1 = resp1.strip()
        resp2 = resp2.strip()

        # Cosine Similarity
        vecs = vectorizer.fit_transform([resp1, resp2])
        cosine = cosine_similarity(vecs[0], vecs[1])[0][0]

        # BLEU
        bleu = sentence_bleu([resp1.split()], resp2.split(), smoothing_function=smoothie)

        # ROUGE
        rouge_score = scorer.score(resp1, resp2)
        rouge_l = rouge_score['rougeL'].fmeasure

        results.append((i+1, cosine, bleu, rouge_l))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Example", "CosineSimilarity", "BLEU", "ROUGE_L"])
        writer.writerows(results)
