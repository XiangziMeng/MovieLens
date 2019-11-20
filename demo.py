#coding: utf-8
import numpy as np
from utils import *

def get_distance(v1, v2):
    return 1 - np.dot(v1, v2) / np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))

#def get_distance(v1, v2):
#    v1 = np.array(v1)
#    v2 = np.array(v2)
#    return np.dot(v1 - v2, v1 - v2)

def get_similar_word(word2embedding, fix_word, word2genres):
    min_distance = 100
    min_word = ""
    fix_embedding = word2embedding[fix_word]
    for word in word2embedding:
        if word == fix_word:
            continue
        embedding = word2embedding[word]
        distance = get_distance(fix_embedding, embedding)
        if distance < min_distance:
            min_distance = distance
            min_word = word
    if min_distance < 100:
        print (word2genres.get(int(fix_word), ""), word2genres.get(int(min_word), ""), min_distance)

if __name__ == "__main__":
    filename = "movie_embeddings.txt"
    index2movie_genres = get_movie_stats("data/movies.dat")
    print (index2movie_genres)
    word2embedding = {}
    for line in open(filename):
        word, raw_embedding = line.strip().split()
        embedding = [float(num) for num in raw_embedding.split(",")]
        word2embedding[word] = embedding
    for i, fix_word in enumerate(word2embedding):
        if i > 200:
            break
        get_similar_word(word2embedding, fix_word, index2movie_genres)

