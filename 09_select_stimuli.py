import itertools
import multiprocessing
import numpy
import os
import pdb
import pickle
import scipy

from scipy import spatial, stats
from tqdm import tqdm

def collect_res(aud_top):
    fourtet_sims = dict()
    for aud_bot in candidates['auditory']['bottom']:
        for act_top in candidates['action']['top']:
            for act_bot in candidates['action']['bottom']:
                combs = itertools.combinations([aud_top, aud_bot, act_top, act_bot], r=2)
                key = tuple(sorted([aud_top, aud_bot, act_top, act_bot]))
                fourtet_sims[key] = list()
                for c in combs:
                    fourtet_sims[key].append(couple_sims[tuple(sorted([c[0], c[1]]))])
    return fourtet_sims

words_and_norms = dict()
with open(os.path.join('output', 'candidate_nouns_all_variables.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line[1:].copy()
            relevant_keys = [h for h in header if 'en_' not in h]
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


### starting to filter: concreteness
conc_ws = [(k, v[relevant_keys.index('predicted_concreteness')]) for k, v in words_and_norms.items()]
sorted_ws = sorted(conc_ws, key=lambda item : item[1], reverse=True)
selected_ws = [w[0] for w in sorted_ws][:int(len(conc_ws)*0.25)]
#selected_ws = [w[0] for w in sorted_ws][:int(len(conc_ws)*0.05)]

### action words
aud_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_auditory')]) for k in selected_ws]
sorted_aud_ws = sorted(aud_ws, key=lambda item : item[1])
bottom_aud_ws = [w[0] for w in sorted_aud_ws][:int(len(aud_ws)*0.25)]
#bottom_aud_ws = [w[0] for w in sorted_aud_ws][int(len(aud_ws)*0.05):]

act_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_hand')]) for k in bottom_aud_ws]
sorted_act_ws = sorted(act_ws, key=lambda item : item[1], reverse=True)
#sel_bottom_act_ws = [w[0] for w in sorted_act_ws][-int(len(act_ws)*0.05):]
#sel_top_act_ws = [w[0] for w in sorted_act_ws][:int(len(act_ws)*0.05)]
sel_bottom_act_ws = [w[0] for w in sorted_act_ws][-int(len(act_ws)*0.25):]
sel_top_act_ws = [w[0] for w in sorted_act_ws][:int(len(act_ws)*0.25)]

### auditory words
act_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_hand')]) for k in selected_ws]
sorted_act_ws = sorted(act_ws, key=lambda item : item[1])
bottom_act_ws = [w[0] for w in sorted_act_ws][:int(len(aud_ws)*0.25)]
top_act_ws = [w[0] for w in sorted_act_ws][-int(len(aud_ws)*0.25):]
#bottom_act_ws = [w[0] for w in sorted_act_ws][int(len(aud_ws)*0.05):]

aud_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_auditory')]) for k in bottom_act_ws]
#aud_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_auditory')]) for k in top_act_ws]
sorted_aud_ws = sorted(aud_ws, key=lambda item : item[1], reverse=True)
#sel_bottom_aud_ws = [w[0] for w in sorted_aud_ws][-int(len(aud_ws)*0.05):]
#sel_top_aud_ws = [w[0] for w in sorted_aud_ws][:int(len(aud_ws)*0.05)]
sel_bottom_aud_ws = [w[0] for w in sorted_aud_ws][-int(len(aud_ws)*0.25):]
sel_top_aud_ws = [w[0] for w in sorted_aud_ws][:int(len(aud_ws)*0.25)]

candidates = {
              'auditory' : {
                            'top' : sel_top_aud_ws.copy(),
                            'bottom' : sel_bottom_aud_ws.copy(),
                            },
              'action' : {
                            'top' : sel_top_act_ws.copy(),
                            'bottom' : sel_bottom_act_ws.copy(),
                            }
              }

remove = list()
sim_words = dict()
for _, v in candidates.items():
    for __, ws in v.items():
        for w in ws:
            vec = numpy.array([words_and_norms[w][h_i] for h_i, h in enumerate(relevant_keys) if h not in remove], dtype=numpy.float64)
            sim_words[w] = vec

couple_sims = dict()
with tqdm() as counter:
    for w_one, vec_one in sim_words.items():
        for w_two, vec_two in sim_words.items():
            #sims[sorted([w_one, w_two])] = scipy.stats.pearsonr(w_one, w_two)[0]
            couple_sims[tuple(sorted([w_one, w_two]))] = 1 - scipy.spatial.distance.cosine(vec_one, vec_two)
            #sims[sorted([w_one, w_two])] = 1 - scipy.spatial.distance.euclidean(w_one, w_two)
            counter.update(1)
with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
    results = pool.map(collect_res, candidates['auditory']['top'])
    pool.terminate()
    pool.join()
all_results = dict()
for res in results:
    for k, v in res.items():
        all_results[k] = v
with open(os.path.join('pickles', 'fourtets_0.05.pkl'), 'wb') as o:
    pickle.dump(all_results, o)
