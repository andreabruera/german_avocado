import itertools
import numpy
import os
import pdb
import pickle
import scipy

from scipy import spatial, stats
from tqdm import tqdm

words_and_norms = dict()
with open(os.path.join('output', 'candidate_nouns_all_variables.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line[1:].copy()
            #relevant_keys = [h for h in header if 'en_' not in h]
            relevant_keys = [h for h in header if 'en_' not in h and 'raw_' not in h]
            continue
        ### filtering words which are too long
        if len(line[0]) > 10:
            continue
        ### filtering compounds
        if len(line[header.index('en_google_translation')+1].split()) > 1:
            continue
        if 'na' in [line[header.index(h)+1] for h in relevant_keys]:
            print(line[0])
            continue
        words_and_norms[line[0]] = [float(line[header.index(h)+1]) for h in relevant_keys]

fourtets = os.path.join('fourtets')
perc = 0.25
words_used = list()
counter = 0
with open(os.path.join(fourtets, 'fourtets_{}.tsv'.format(perc))) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')[:4]
        if l_i == 0:
            mapper = {h_i : h for h_i, h in enumerate(line)}
            fourtets = {h : list() for h in line}
            continue
        check = [False if w not in words_used else True for w in line]
        if True in check:
            continue
        counter += 1
        if counter > 30:
            continue
        words_used.extend(line)
        for i in range(4):
            fourtets[mapper[i]].append(line[i])

### testing p-vals
combs = list(itertools.combinations(fourtets.keys(), r=2))
for h in relevant_keys:
    vals = {k : [words_and_norms[w][relevant_keys.index(h)] for w in v] for k, v in fourtets.items()}
    ps = list()
    for c in combs:
        p = scipy.stats.ttest_ind(vals[c[0]], vals[c[1]]).pvalue
        ps.append(p)
    check = [False if p>0.05 else True for p in ps]
    print(h)
    print(ps)
    if True in check:
        print(h)
