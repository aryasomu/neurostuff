from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def rank_resumes(resume_texts, job_description):
    corpus = resume_texts + [job_description]
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    job_vec = vectors[-1]
    resume_vecs = vectors[:-1]
    similarities = cosine_similarity(resume_vecs, job_vec)
    ranked_indices = np.argsort(similarities.flatten())[::-1]
    return ranked_indices, similarities.flatten()[ranked_indices]