import itertools
import matplotlib
import mne
import numpy
import os
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
            variables = {h : dict() for h in line[1:]}
        else:
            for h, val in zip(header, line):
                if h == 'word':
                    word = val
                    full_words.append(word)
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
          ]
relevant_keys = [h for h in variables.keys() if 'en_' not in h and 'raw_' not in h and h not in remove and 'proto' not in h]

amount_stim = 36

### reading words from previous experiments
old_file = os.path.join('output', 'phil_original_annotated_clean.tsv')
old_goods = dict()
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
        old_goods[label].add(line[word])
print([(k, len(v)) for k, v in old_goods.items()])

for mode in ('good_only', 'good_mid'):

    ### reading selected nouns
    good = {l : v for l, v in old_goods.items()}
    localizer = {
                 'action' : set(), 
                 'sound' : set(),
                 }

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
                        localizer[line[1]].add(line[0])
                    else:
                        good[label].add(line[0])
                else:
                    if line[1] in ['bad']:
                        continue
                    elif line[1] in ['action', 'sound']:
                        localizer[line[1]].add(line[0])
                    else:
                        good[label].add(line[0])
    print('good items')
    print([(k, len(v)) for k, v in good.items()])
    print('localizer items')
    print([(k, len(v)) for k, v in localizer.items()])
    ### plotting distributions

    ### plotting violinplots
    violin_folder = os.path.join('violins', mode)
    os.makedirs(violin_folder, exist_ok=True)
    xs = [val for val in good.keys()]
    for k in relevant_keys:
        print(k)
        file_name = os.path.join(violin_folder, '{}.jpg'.format(k))
        fig, ax = pyplot.subplots(constrained_layout=True)
        for _ in range(len(xs)):
            ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in good[xs[_]]], showmeans=False)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([x.replace('_', '_') for x in xs])
        ax.set_title('{} distributions for selected words'.format(k))
        pyplot.savefig(file_name)
        pyplot.clf()
        pyplot.close()

### propose selection of stimuli
### compute averages for each condition

idxs = [var for var in relevant_keys if 'hand' not in var and 'auditory' not in var]
exp_idxs = [var for var in relevant_keys if 'hand' in var or 'auditory' in var]
variable_avgs = {var: numpy.average([float(variables[var][w]) for v in good.values() for w in v]) for var in idxs}
exp_avgs = {var: numpy.average([float(variables[var][w]) for v in good.values() for w in v]) for var in exp_idxs}
distances = {w : list() for v in good.values() for w in v}
for _, v in good.items():
    for w in v:
        for var, var_avg in variable_avgs.items():
            if 'concreteness' not in var:
                distances[w].append(abs(float(variables[var][w])-var_avg))
            else:
                if 'lowA' in _:
                    distances[w].append(var_avg-float(variables[var][w]))
                elif 'highA' in _:
                    distances[w].append(-float(variables[var][w])+var_avg)

        if 'lowS' in _:
            distances[w].append(abs(exp_avgs['predicted_auditory']-float(variables['predicted_auditory'][w])))
        elif 'highS' in _:
            distances[w].append(abs(float(variables['predicted_auditory'][w])-exp_avgs['predicted_auditory']))
        if 'lowA' in _:
            distances[w].append(abs(exp_avgs['predicted_hand']-float(variables['predicted_hand'][w])))
        elif 'highA' in _:
            distances[w].append(abs(float(variables['predicted_hand'][w])-exp_avgs['predicted_hand']))
distances = {k : numpy.average(v) for k, v in distances.items()}

best_good = {label : {w : distances[w] for w in v} for label, v in good.items()}
best_good = {label : [w[0] for w in sorted(v.items(), key=lambda item : item[1])] for label, v in best_good.items()}

### plotting violinplots
violin_folder = os.path.join('violins', 'best_for_experiment')
os.makedirs(violin_folder, exist_ok=True)
xs = [val for val in best_good.keys()]
### testing p-vals
ps = list()
cases = list()
combs = list(itertools.combinations(xs, r=2))
for k in relevant_keys:
    vals = {xs[_] : [float(variables[k][w]) for w in best_good[xs[_]]][:amount_stim] for _ in range(len(xs))}
    for c in combs:
        p = scipy.stats.ttest_ind(vals[c[0]], vals[c[1]]).pvalue
        ps.append(p)
        cases.append([k, c[0], c[1]])
    file_name = os.path.join(violin_folder, '{}.jpg'.format(k))
    fig, ax = pyplot.subplots(constrained_layout=True)
    for _ in range(len(xs)):
        ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in best_good[xs[_]][:amount_stim]], showmeans=True)
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
        ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in best_good[xs[_]][amount_stim:]], showmeans=True)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([x.replace('_', '_') for x in xs])
    ax.set_title('{} distributions for selected words'.format(k))
    pyplot.savefig(file_name)
    pyplot.clf()
    pyplot.close()