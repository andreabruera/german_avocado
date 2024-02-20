import itertools
import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import pyplot
from scipy import stats

### reading ratings
full_words = list()
with open(os.path.join('output', 'candidate_nouns_all_variables.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            variables = {h : dict() for h in line[1:] if 'lemma' not in h and h not in ['predicted_dominance', 'predicted_arousal', 'predicted_valence']}
        else:
            for h, val in zip(header, line):
                if h == 'word':
                    word = val
                    full_words.append(word)
                elif 'lemma' in h:
                    continue
                elif h in ['predicted_dominance', 'predicted_arousal', 'predicted_valence']:
                    continue
                else:
                    if val.isnumeric():
                        val = float(val)
                        if 'gram' in h:
                            val = numpy.log10(val)
                    variables[h][word] = val

remove = [
          'predicted_leg', 
          'predicted_haptic', 
          'predicted_mouth', 
          'predicted_head', 
          'predicted_torso', 
          'predicted_visual',
          'predicted_olfactory',
          'predicted_gustatory',
          'predicted_hearing',
          ]
relevant_keys = [h for h in variables.keys() if 'en_' not in h and 'raw_' not in h and h not in remove and 'proto' not in h]

amount_stim = 36

### reading words from previous experiments
old_file = os.path.join('output', 'phil_original_annotated_clean.tsv')
old_goods = dict()
localizer = {
             'low_action' : set(), 
             'low_sound' : set(),
             'high_action' : set(), 
             'high_sound' : set(),
             }
old_localizer = dict()
with open(old_file) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            included = line.index('Included')
            action = line.index('Action')
            sound = line.index('Sound')
            word = line.index('Words')
            continue
        eval_val = float(line[included].replace(',', '.'))
        #print(eval_val)
        assert eval_val >= 0. and eval_val <=1.
        if eval_val == 0.:
            continue
        curr_action = float(line[action].replace(',', '.'))
        curr_sound = float(line[sound].replace(',', '.'))
        assert curr_action in [0., 1.] and curr_sound in [0., 1.]
        if curr_action > 0. and curr_sound > 0.:
            label = 'highA_highS'
        if curr_action == 0. and curr_sound == 0.:
            label = 'lowA_lowS'
        if curr_action == 0. and curr_sound > 0.:
            label = 'lowA_highS'
        if curr_action > 0. and curr_sound == 0.:
            label = 'highA_lowS'
        if label not in old_goods.keys():
            old_goods[label] = set()
        if line[word] not in full_words:
            print(line[word])
            continue
        if eval_val < 0.9:
            if eval_val > 0.5:
                if 'highA' in label:
                    localizer['high_action'].add(line[word])
                if 'highS' in label:
                    localizer['high_sound'].add(line[word])
                if 'lowA' in label:
                    localizer['low_action'].add(line[word])
                if 'lowS' in label:
                    localizer['low_sound'].add(line[word])
        else:
            old_goods[label].add(line[word])
print('old items')
print([(k, len(v)) for k, v in old_goods.items()])
print('old localizers')
print([(k, len(v)) for k, v in localizer.items()])

### testing p-vals
ps = list()
cases = list()

for mode in (
             'good_only', 
             #'original_exp',
             #'good_mid',
             ):

    ### reading selected nouns
    good = {l : v for l, v in old_goods.items()}
    if mode != 'original_exp':


        folder = 'Stimuli_annotated'
        for f in os.listdir(folder):
            if 'tsv' not in f:
                continue
            ### category
            label = '_'.join(f.split('_')[:2])
            #if label not in good.keys():
            #    good[label] = set()
            with open(os.path.join(folder, f)) as i:
                for l_i, l in enumerate(i):
                    if l_i == 0:
                        continue
                    line = l.strip().split('\t')
                    if line[0] == 'word':
                        continue
                    if mode == 'good_only':
                        if line[1] in ['mid', 'bad']:
                            continue
                        elif line[1] in ['action', 'sound']:
                            if 'highS' in f:
                                localizer['high_sound'].add(line[0])
                            if 'lowS' in f:
                                localizer['low_sound'].add(line[0])
                            if 'highA' in f:
                                localizer['high_action'].add(line[0])
                            if 'lowA' in f:
                                localizer['low_action'].add(line[0])
                        else:
                            good[label].add(line[0])
                    else:
                        if line[1] in ['bad']:
                            continue
                        elif line[1] in ['action', 'sound']:
                            if 'highS' in f:
                                localizer['high_sound'].add(line[0])
                            if 'lowS' in f:
                                localizer['low_sound'].add(line[0])
                            if 'highA' in f:
                                localizer['high_action'].add(line[0])
                            if 'lowA' in f:
                                localizer['low_action'].add(line[0])
                        else:
                            good[label].add(line[0])
        print('localizer items')
        print([(k, len(v)) for k, v in localizer.items()])
    ### plotting distributions
    print('good items')
    print([(k, len(v)) for k, v in good.items()])
    print('localizer items')
    print([(k, len(v)) for k, v in localizer.items()])

    ### plotting violinplots
    violin_folder = os.path.join('violins', mode)
    os.makedirs(violin_folder, exist_ok=True)
    xs = [val for val in good.keys()]
    for k in relevant_keys:
        #print(k)
        if mode == 'original_exp':
            xs = [val for val in good.keys()]
            combs = list(itertools.combinations(xs, r=2))
            vals = {xs[_] : [float(variables[k][w]) for w in good[xs[_]]] for _ in range(len(xs))}
            for c in combs:
                p = scipy.stats.ttest_ind(vals[c[0]], vals[c[1]]).pvalue
                ps.append(p)
                cases.append([k, c[0], c[1]])
        file_name = os.path.join(violin_folder, '{}.jpg'.format(k))
        fig, ax = pyplot.subplots(constrained_layout=True)
        for _ in range(len(xs)):
            ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in good[xs[_]]], showmeans=True)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([x.replace('_', '_') for x in xs])
        ax.set_title('{} distributions for selected words'.format(k))
        pyplot.savefig(file_name)
        pyplot.clf()
        pyplot.close()
corrected_ps = mne.stats.fdr_correction(ps)[1]
#for case, p in zip(cases, corrected_ps):
for case, p in zip(cases, ps):
    if p<=0.05:
        print([case, p])
#print(k)

### propose selection of stimuli
'''
### compute averages for each condition

idxs = [var for var in relevant_keys if 'hand' not in var and 'auditory' not in var]
exp_idxs = [var for var in relevant_keys if 'hand' in var or 'auditory' in var]
distances = {w : list() for v in good.values() for w in v}
### criterion: average across all
variable_avgs = {var: numpy.average([float(variables[var][w]) for k, v in good.items() for w in v]) for var in idxs}
exp_avgs = {var: numpy.average([float(variables[var][w]) for k, v in good.items() for w in v]) for var in exp_idxs}
for _, v in good.items():
    for w in v:
        for var, var_avg in variable_avgs.items():
            distances[w].append(abs(float(variables[var][w])-var_avg))

        if 'lowS' in _:
            distances[w].append(abs(exp_avgs['predicted_auditory']-float(variables['predicted_auditory'][w])))
        elif 'highS' in _:
            distances[w].append(abs(float(variables['predicted_auditory'][w])-exp_avgs['predicted_auditory']))
        if 'lowA' in _:
            distances[w].append(abs(exp_avgs['predicted_hand']-float(variables['predicted_hand'][w])))
        elif 'highA' in _:
            distances[w].append(abs(float(variables['predicted_hand'][w])-exp_avgs['predicted_hand']))
distances = {k : numpy.average(v) for k, v in distances.items()}
'''
distances = dict()
for k, v in good.items():
    split_k = k.split('_')
    ### every word
    for w in v:
        distances[w] = list()
        ### the thing we really care about are hand and audition
        for var_i, var in enumerate(['predicted_hand', 'predicted_auditory']):
            rel_keys = [k for k in good.keys() if split_k[var_i] in k]
            rel_vals = [float(variables[var][w_two]) for key in rel_keys for  w_two in good[key]]
            dist = (numpy.average(rel_vals)**2)-(float(variables[var][w])**2)
            distances[w].append(dist)

best_good = {label : {w : distances[w] for w in v} for label, v in good.items()}
best_good = {label : [w[0] for w in sorted(v.items(), key=lambda item : item[1])] for label, v in best_good.items()}
### criterion: average separately for high/low action/sound
best_good = {k : random.sample(list(v), k=len(v)) for k, v in good.items()}
for v in best_good.values():
    assert len(v) >= amount_stim*2

### plotting violinplots
violin_folder = os.path.join('violins', 'best_for_experiment')
os.makedirs(violin_folder, exist_ok=True)
xs = [val for val in best_good.keys()]
### testing p-vals
ps = list()
cases = list()
combs = list(itertools.combinations(xs, r=2))
for k in relevant_keys:
    vals = {xs[_] : [float(variables[k][w]) for w in best_good[xs[_]]][:amount_stim*2] for _ in range(len(xs))}
    for c in combs:
        p = scipy.stats.ttest_ind(vals[c[0]], vals[c[1]]).pvalue
        ps.append(p)
        cases.append([k, c[0], c[1]])
    file_name = os.path.join(violin_folder, '{}.jpg'.format(k))
    fig, ax = pyplot.subplots(constrained_layout=True)
    for _ in range(len(xs)):
        ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in best_good[xs[_]][:amount_stim*2]], showmeans=True)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([x.replace('_', '_') for x in xs])
    ax.set_title('{} distributions for selected words'.format(k))
    pyplot.savefig(file_name)
    pyplot.clf()
    pyplot.close()

corrected_ps = mne.stats.fdr_correction(ps)[1]
#for case, p in zip(cases, ps):
for case, p in zip(cases, corrected_ps):
    if p<=0.05:
        print([case, p])
print(k)

### plotting violinplots
violin_folder = os.path.join('violins', 'left_for_localizer')
os.makedirs(violin_folder, exist_ok=True)
xs = [val for val in best_good.keys()]
for k in relevant_keys:
    print(k)
    file_name = os.path.join(violin_folder, '{}.jpg'.format(k))
    fig, ax = pyplot.subplots(constrained_layout=True)
    for _ in range(len(xs)):
        ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in best_good[xs[_]][amount_stim*2:]], showmeans=True)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([x.replace('_', '_') for x in xs])
    ax.set_title('{} distributions for selected words'.format(k))
    pyplot.savefig(file_name)
    pyplot.clf()
    pyplot.close()
