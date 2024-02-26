import multiprocessing
import numpy
import os
import pickle
import re

from tqdm import tqdm

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

def multiprocessing_levenshtein(w):
    levs = list()
    for w_two in other_words:
        if w_two == w:
            continue
        levs.append(levenshtein(w, w_two))
    score = numpy.average(sorted(levs)[:20])
    return (w, score)

'''
### reading phil's annotated dataset
relevant_words = list(set(dataset['word']))
#print(relevant_words)
relevant_words = list()
with open(os.path.join('data', 'phil_original_annotated_clean.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            rel_idx = line.index('Words')
            continue
        relevant_words.append(line[rel_idx])
'''
### reading candidate dataset
relevant_words = list()
with open(os.path.join('output', 'candidate_nouns_min_100.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            continue
        relevant_words.append(line[0])

with open(os.path.join('pickles', 'sdewac_lemma_freqs.pkl'), 'rb') as i:
    all_sdewac_freqs = pickle.load(i)

all_sdewac_freqs = {k : v for k, v in all_sdewac_freqs.items() if len(k)>1 and len(re.findall('\||@|<|>',k))==0}
### number of words in the original OLD20 paper
max_n = 35502
global other_words
other_words = [w[0] for w in sorted(all_sdewac_freqs.items(), key=lambda item : item[1], reverse=True)][:max_n]
#print(other_words[-1])

def print_stuff(inputs):
    print(inputs)

'''
old20_scores = {w : 0 for w in relevant_words}
for w in tqdm(relevant_words):
    _, score = multiprocessing_levenshtein(w)
    print([w, score])
    old20_scores[w] = score
'''
with multiprocessing.Pool() as i:
    res = i.map(multiprocessing_levenshtein, relevant_words)
    i.terminate()
    i.join()
for w, val in res:
    old20_scores[w] = val

#with open('old20_scores.tsv', 'w') as o:
with open(os.path.join('output', 'candidate_nouns_old20.tsv'), 'w') as o:
    o.write('word\told20 score (based on the top {} lemmas in sdewac)\n'.format(max_n))
    for k, v in old20_scores.items():
        o.write('{}\t{}\n'.format(k, v))
