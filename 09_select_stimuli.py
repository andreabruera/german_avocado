import itertools
import multiprocessing
import numpy
import os
import pdb
import pickle
import scipy

from scipy import spatial, stats
from tqdm import tqdm

def collect_res(possibility):
    fourtet_sims = dict()
    combs = itertools.combinations(possibility, r=2)
    key = tuple(sorted(possibility))
    #fourtet_sims[key] = list()
    fourtet_sims = list()
    for c in combs:
        #fourtet_sims[key].append(couple_sims[tuple(sorted([c[0], c[1]]))])
        fourtet_sims.append(couple_sims[tuple(sorted([c[0], c[1]]))])
    return key, fourtet_sims

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
# keep only top 25% most concrete nouns
selected_ws = [w[0] for w in sorted_ws][:int(len(conc_ws)*0.25)]
message = 'retaining {} most concrete words'.format(len(selected_ws))
print(message)

#perc = 0.1
#perc = 0.15
#perc = 0.2
#perc = 0.25
perc = 0.05

### selecting the four corners
corners = {
           'auditory' :
                   {
                    'top' : list(),
                    'bottom' : list(),
                    },
           'action' :
                   {
                    'top' : list(),
                    'bottom' : list(),
                    },
           }
## auditory words
aud_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_auditory')]) for k in selected_ws]
sorted_aud_ws = sorted(aud_ws, key=lambda item : item[1])
corners['auditory']['bottom'].extend([w[0] for w in sorted_aud_ws][:int(len(aud_ws)*perc)])
corners['auditory']['top'].extend([w[0] for w in sorted_aud_ws][-int(len(aud_ws)*perc):])
## action words
act_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_hand')]) for k in selected_ws]
sorted_act_ws = sorted(act_ws, key=lambda item : item[1])
corners['action']['bottom'].extend([w[0] for w in sorted_act_ws][:int(len(act_ws)*perc)])
corners['action']['top'].extend([w[0] for w in sorted_act_ws][-int(len(act_ws)*perc):])
checks = [len(v) for _ in corners.values() for v in _.values()]
poss = set(checks)
assert len(poss) == 1
print('for each corner, {} possibilities'.format(poss))

### 
mapper = {
          'action' : 'predicted_hand',
          'auditory' : 'predicted_auditory',
          }
candidates = dict()
for k_one, v_one in corners.items():
    for k_two, v_two in corners.items():
        if k_one == k_two:
            continue
        for side_one, ws_one in v_one.items():
            key_one = '{}_{}'.format(k_one, side_one)
            ws = [(k, words_and_norms[k][relevant_keys.index(mapper[k_two])]) for k in ws_one]
            sorted_ws = sorted(ws, key=lambda item : item[1])
            ### bottom
            key_two = '{}_bottom'.format(k_two)
            cand_key = tuple(sorted([key_one, key_two]))
            if cand_key not in candidates.keys():
                candidates[cand_key] = set()
            candidates[cand_key].update(set([w[0] for w in sorted_ws[:int(len(sorted_ws)*perc)]]))
            ### top 
            key_two = '{}_top'.format(k_two)
            cand_key = tuple(sorted([key_one, key_two]))
            if cand_key not in candidates.keys():
                candidates[cand_key] = set()
            candidates[cand_key].update(set([w[0] for w in sorted_ws[-int(len(sorted_ws)*perc):]]))

### writing candidates to file
cand_out = 'candidates'
os.makedirs(cand_out, exist_ok=True)
for k, v in candidates.items():
    f_k = '_'.join(k)
    with open(os.path.join(cand_out, '{}_{}.txt'.format(f_k, perc)), 'w') as o:
        for w in v:
            o.write('{}\n'.format(w))

### putting it all together

remove = list()
sim_words = dict()
for __, ws in candidates.items():
    for w in ws:
        vec = numpy.array([words_and_norms[w][h_i] for h_i, h in enumerate(relevant_keys) if h not in remove], dtype=numpy.float64)
        sim_words[w] = vec

couple_sims = dict()
with tqdm() as counter:
    for _, ws_one in candidates.items():
        for __, ws_two in candidates.items():
            if _ == __:
                continue
            for w_one in ws_one:
                for w_two in ws_two:
                    couple_sims[tuple(sorted([w_one, w_two]))] = 1 - scipy.spatial.distance.cosine(sim_words[w_one], sim_words[w_two])
                    counter.update(1)
p = [v for v in candidates.values()]
possibilities = itertools.product(p[0], p[1], p[2], p[3], repeat=1)

res = dict()
with tqdm() as counter:
    for p in possibilities:
        k, v = collect_res(p)
        res[k] = v
        counter.update(1)
#with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
#    results = pool.map(collect_res, possibilities)
#    pool.terminate()
#    pool.join()
#all_results = dict()
#for res in results:
#    for k, v in res.items():
#        all_results[k] = v
with open(os.path.join('pickles', 'fourtets_{}.pkl'.format(perc)), 'wb') as o:
    pickle.dump(res, o)
