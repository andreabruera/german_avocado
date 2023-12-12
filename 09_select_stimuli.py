import itertools
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
            relevant_keys = [h for h in header if 'en_' not in h and 'raw_' not in h]
            continue
        ### filtering words which are too long
        if len(line[0]) > 12:
            continue
        ### filtering compounds
        if len(line[header.index('en_google_translation')+1].split()) > 1:
            continue
        if 'nan' in [line[header.index(h)+1] for h in relevant_keys]:
            print(line)
            continue
        if 'na' in [line[header.index(h)+1] for h in relevant_keys]:
            print(line)
            continue
        words_and_norms[line[0]] = [float(line[header.index(h)+1]) for h in relevant_keys]
        for dim in words_and_norms[line[0]]:
            assert str(dim) != 'nan'


### starting to filter: concreteness
conc_ws = [(k, v[relevant_keys.index('predicted_concreteness')]) for k, v in words_and_norms.items()]
sorted_ws = sorted(conc_ws, key=lambda item : item[1], reverse=True)
# keep only top 25% most concrete nouns
selected_ws = [w[0] for w in sorted_ws][:int(len(conc_ws)*0.25)]
message = 'retaining {} most concrete words'.format(len(selected_ws))
print(message)
remove = [
          'predicted_leg', 
          'predicted_haptic', 
          'predicted_mouth', 
          'predicted_head', 
          'predicted_torso', 
          ]

bottom_perc = 0.5
for top_perc in [
             0.05, 
             #0.1, 
             #0.15, 
             #0.2, 
             #0.25,
             #0.33
             ]:

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
    corners['auditory']['bottom'].extend([w[0] for w in sorted_aud_ws][:int(len(aud_ws)*bottom_perc)])
    corners['auditory']['top'].extend([w[0] for w in sorted_aud_ws][-int(len(aud_ws)*top_perc):])
    ## action words
    act_ws = [(k, words_and_norms[k][relevant_keys.index('predicted_hand')]) for k in selected_ws]
    sorted_act_ws = sorted(act_ws, key=lambda item : item[1])
    corners['action']['bottom'].extend([w[0] for w in sorted_act_ws][:int(len(act_ws)*bottom_perc)])
    corners['action']['top'].extend([w[0] for w in sorted_act_ws][-int(len(act_ws)*top_perc):])
    checks = [len(v) for _ in corners.values() for v in _.values()]
    poss = set(checks)
    #assert len(poss) == 1
    print('for each corner, {} possibilities'.format(poss))
    averages = {k : {c : numpy.median([words_and_norms[v][relevant_keys.index('predicted_hand')] if k=='action' else words_and_norms[v][relevant_keys.index('predicted_auditory')] for v in ws]) for c, ws in kv.items()} for k, kv in corners.items()}

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
                diffs = [(k, abs(averages[k_two]['bottom']-v)) for k, v in ws]
                sorted_diffs = sorted(ws, key=lambda item : item[1])
                #candidates[cand_key].update(set([w[0] for w in sorted_ws[:int(len(sorted_ws)*bottom_perc)]]))
                candidates[cand_key].update(set([w[0] for w in sorted_diffs[:int(len(sorted_diffs)*bottom_perc)]]))
                ### top 
                key_two = '{}_top'.format(k_two)
                cand_key = tuple(sorted([key_one, key_two]))
                if cand_key not in candidates.keys():
                    candidates[cand_key] = set()
                diffs = [(k, abs(averages[k_two]['top']-v)) for k, v in ws]
                sorted_diffs = sorted(ws, key=lambda item : item[1])
                #candidates[cand_key].update(set([w[0] for w in sorted_ws[-int(len(sorted_ws)*top_perc):]]))
                candidates[cand_key].update(set([w[0] for w in sorted_diffs[-int(len(sorted_diffs)*top_perc):]]))
    word_to_key = {w : k for k, v in candidates.items() for w in v}

    ### writing candidates to file
    cand_out = os.path.join('candidates', str(top_perc))
    os.makedirs(cand_out, exist_ok=True)
    for k, v in candidates.items():
        f_k = '_'.join(k)
        with open(os.path.join(cand_out, '{}_{}.txt'.format(f_k, top_perc)), 'w') as o:
            o.write('word\tgood_or_bad?\n')
            for w in v:
                o.write('{}\tx\n'.format(w))

    ### putting it all together

    word_vecs = dict()
    for __, ws in candidates.items():
        for w in ws:
            vec = numpy.array([words_and_norms[w][h_i] for h_i, h in enumerate(relevant_keys)], dtype=numpy.float64)
            word_vecs[w] = vec
    for v in word_vecs.values():
        for dim in v:
            assert str(dim) != 'nan'
    ### z-scoring
    means = numpy.average([v for v in word_vecs.values()], axis=0)
    stds = numpy.std([v for v in word_vecs.values()], axis=0)
    word_vecs = {w : (v-means)/stds for w, v in word_vecs.items()}
    for v in word_vecs.values():
        for dim in v:
            assert str(dim) != 'nan'

    couple_sims = dict()
    with tqdm() as counter:
        for _, ws_one in candidates.items():
            for __, ws_two in candidates.items():
                if _ == __:
                    continue
                if 'auditory_bottom' in _ and 'auditory_bottom' in __:
                    current_remove = remove + ['predicted_hand']
                if 'auditory_top' in _ and 'auditory_top' in __:
                    current_remove = remove + ['predicted_hand']
                if 'action_bottom' in _ and 'action_bottom' in __:
                    current_remove = remove + ['predicted_auditory']
                if 'action_top' in _ and 'action_top' in __:
                    current_remove = remove + ['predicted_auditory']
                weights = [0.25 if 'predicted' not in h else 1. for h in relevant_keys if h not in current_remove]
                for w_one in ws_one:
                    vec_one = [dim for dim_i, dim in enumerate(word_vecs[w_one]) if relevant_keys[dim_i] not in current_remove]
                    for w_two in ws_two:
                        vec_two = [dim for dim_i, dim in enumerate(word_vecs[w_two]) if relevant_keys[dim_i] not in current_remove]
                        couple_sims[tuple(sorted([w_one, w_two]))] = 1 - scipy.spatial.distance.cosine(vec_one, vec_two, w=weights)
                        counter.update(1)

    # we use each word as a seed to get possibilities
    # for each word, we do not consider more than 10 possibilities
    words_used_counter = {w : 0 for v in candidates.values() for w in v}
    fourtets = list()
    with tqdm() as counter:
        for k_one, ws_one in candidates.items():
            for w_one in ws_one:
                #if words_used_counter[w_one]>=100:
                #    continue
                chosen_ws = [[w_one] for i in range(10)]
                chosen_sims = [list() for i in range(10)]
                for k_two, ws_two in candidates.items():
                    if k_one == k_two:
                        continue
                    #clean_ws_two = [w for w in ws_two if words_used_counter[w]<100]
                    clean_ws_two = [w for w in ws_two]
                    if len(clean_ws_two) == 0:
                        continue
                    sims = [(w, couple_sims[tuple(sorted([w, w_one]))]) for w in clean_ws_two]
                    for i in range(5):
                        chosen_w = sorted(sims, key=lambda item : item[1], reverse=True)[i]
                        chosen_ws[i].append(chosen_w[0])
                        chosen_sims[i].append(chosen_w[1])
                for i in range(5):
                    if len(chosen_ws[i]) < 4:
                        continue
                    for w in chosen_ws[i]:
                        words_used_counter[w] += 1
                    fourtets.append(chosen_ws[i]+[numpy.average(chosen_sims[i]), numpy.std(chosen_sims[i])])
                counter.update(1)
    ### sorting and reordering
    columns = [k for k in candidates.keys()]
    sorted_fourtets = sorted(fourtets, key=lambda item : item[-2], reverse=True)
    reordered_fourtets = list()
    for line in sorted_fourtets:
        words = line[:4]
        reordered_words = list()
        for col in columns:
            for w in words:
                if word_to_key[w] == col:
                    reordered_words.append(w)
        assert len(reordered_words) == 4
        reordered_line = reordered_words + line[4:]
        assert len(reordered_line) == 6
        reordered_fourtets.append(reordered_line)
    
    out_fourtets = os.path.join('fourtets')
    os.makedirs(out_fourtets,exist_ok=True)
    with open(os.path.join(out_fourtets, 'fourtets_{}.tsv'.format(top_perc)), 'w') as o:
        for col in columns:
            o.write('{}\t'.format(col))
        o.write('average_similarity\taverage_std\n')
        for line in reordered_fourtets:
            for val in line:
                o.write('{}\t'.format(val))
            o.write('\n')
