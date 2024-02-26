import itertools
import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import pyplot
from scipy import stats

def skip_words(word, label):
    marker = False
    '''
    if label == 'highA_lowS':
        if float(variables['predicted_concreteness'][word]) > 2.:
            marker = True
    ### this is where we can cut the most...
    if label == 'lowA_lowS':
        if float(variables['predicted_concreteness'][word]) < 1.1:
            marker = True
        if float(variables['word_average_trigram_frequency'][word]) > 10500000:
            marker = True
        if float(variables['log10_word_frequency_sdewac'][word]) > 3.9:
                marker = True
    if 'lowA_lowS' in label:
        if float(variables['predicted_auditory'][word]) > -.75:
            marker = True
        if float(variables['predicted_concreteness'][word]) > 3.:
            marker = True
    if 'highA_lowS' in label:
        if float(variables['predicted_concreteness'][word]) > 2.5:
            marker = True
    if 'highS' in label:
        if float(variables['predicted_concreteness'][word]) > 3.:
            marker = True
    '''
    #print(label)
    #if 'highS' in label:
        #if float(variables['word_length'][word]) < 5:
        #    marker = True
        #if float(variables['word_length'][word]) > 11:
        #    marker = True
    '''
        if float(variables['predicted_auditory'][word]) > 0.:
            marker = True
    if 'lowA' in label:
        if float(variables['predicted_auditory'][word]) >= 0.:
            marker = True
    '''
    if 'lowS' in label:
        if float(variables['predicted_concreteness'][word]) > 2.5:
            marker = True
        if float(variables['word_length'][word]) <= 5:
            marker = True
        if float(variables['word_average_trigram_frequency'][word]) > 1.1*1e7:
            marker = True
    if 'highA' in label:
        if float(variables['predicted_concreteness'][word]) > 2.5:
            marker = True
        if float(variables['predicted_concreteness'][word]) < .75:
            marker = True
    if 'lowA' in label:
        if float(variables['predicted_concreteness'][word]) < 1.5:
            marker = True
        if float(variables['word_average_bigram_frequency'][word]) < .5*1e7:
            marker = True
        if float(variables['word_length'][word]) < 5:
            marker = True
    if word in selected_words:
        marker = True
    if word in corrections:
        marker = True
    if word in localizer_corrections[label]:
        print(word)
        marker = True
    return marker

### reading selected words
global selected_words
selected_words = list()
with open(os.path.join('txt_results', 'good_only', '48', 'main_experiment_words_good_only_48.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        selected_words.append(line[0])
assert len(selected_words) == 48*2*4

### reading ratings
full_words = list()
with open(os.path.join('output', 'candidate_nouns_all_variables.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = line.copy()
            variables = {h : dict() for h in line[1:] if 'lemma' not in h}
        else:
            for h, val in zip(header, line):
                if h == 'word':
                    word = val
                    full_words.append(word)
                elif 'lemma' in h:
                    continue
                else:
                    if val.isnumeric():
                        val = float(val)
                        if 'gram' in h:
                            print(h)
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
          'predicted_valence',
          'predicted_dominance',
          'predicted_arousal',
          ]
relevant_keys = [h for h in variables.keys() if 'en_' not in h and 'raw_' not in h and h not in remove and 'proto' not in h]

### reading latest corrections
global corrections
corrections = list()
with open('phil_correction_21_02.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        if len(line) > 2:
            if line[2].strip() in ['bad']:
                corrections.append(line[2].strip())
### reading latest corrections
global localizer_corrections
localizer_corrections = dict()
with open('localizer_bad.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        if len(line) > 2:
            if line[2].strip() in ['mid', 'bad']:
                if line[1] not in localizer_corrections.keys():
                    localizer_corrections[line[1]] = list()
                localizer_corrections[line[1]].append(line[0].strip())

### reading words from previous experiments
old_file = os.path.join('output', 'phil_original_annotated_clean.tsv')
localizer = dict()
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
        if label not in localizer.keys():
            localizer[label] = set()
        marker = False
        ### words from the original experiment that are missing...
        if line[word] not in full_words:
            print(line[word])
            continue
        ### pruning distributions
        #if skip_words(line[word], label) == True:
        #    continue
        ### checking evaluation is high enough, else localizer
        if eval_val > 0.5:
            localizer[label].add(line[word])

print('old localizers')
print([(k, len(v)) for k, v in localizer.items()])

folder = 'Stimuli_annotated'
for f in os.listdir(folder):
    if 'tsv' not in f:
        continue
    ### category
    label = '_'.join(f.split('_')[:2])
    with open(os.path.join(folder, f)) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            if line[0] == 'word':
                continue
            ### pruning distributions
            #if skip_words(line[0], label) == True:
            #    continue
            ### dividing into bad/mid/good
            if line[1] in ['bad']:
                continue
            elif line[1] in ['good', 'mid', 'action' 'sound']:
                localizer[label].add(line[0])
print('localizer items')
print([(k, len(v)) for k, v in localizer.items()])

localizer_words = {
                   'lowA' : list(),
                   'highA' : list(),
                   'lowS' : list(),
                   'highS' : list(),
                   }
localizer_words = {k : [w for _, v in localizer.items() for w in v if k in _] for k in localizer_words.keys()}

### removing wrong ends of the distributions
localizer_words = {k : [w for w in v if skip_words(w, k)==False] for k,v in localizer_words.items()}
print('localizer items')
print([(k, len(v)) for k, v in localizer_words.items()])

distances = dict()
for split_k, v in localizer_words.items():
    ### every word
    for w in v:
        distances[w] = list()
        '''
        ### the thing we really care about are hand and audition
        for var_i, var in enumerate([
                                     'predicted_hand', 
                                     'predicted_auditory',
                                     ]):
            ### promoting similarity
            rel_keys = [split_k]
            #assert len(rel_keys) == 2
            assert len(rel_keys) == 1
            all_vars = [
                       var, 
                       #'predicted_concreteness'
                       ]
            #all_vars = [var]
            for all_var in all_vars:
                rel_vals = [float(variables[all_var][w_two]) for key in rel_keys for w_two in localizer_words[key]]
                rel_avg = numpy.average(rel_vals)
                dist = abs(rel_avg-float(variables[all_var][w]))
                distances[w].append(dist)
        '''
        ### promoting similarity to the other cat
        rel_keys = [k for k in localizer_words.keys() if k[-1]==split_k[-1] and k!=split_k]
        #assert len(rel_keys) == 2
        assert len(rel_keys) == 1
        all_vars = [
                   #var, 
                   'predicted_concreteness',
                   #'old20_score',
                   ]
        #all_vars = [var]
        for all_var in all_vars:
            rel_vals = [float(variables[all_var][w_two]) for key in rel_keys for w_two in localizer_words[key]]
            rel_avg = numpy.average(rel_vals)
            dist = abs(rel_avg-float(variables[all_var][w]))
            distances[w].append(dist)
        '''
        ### promoting dissimilarity
        unrel_keys = [split_k.replace('low', 'high')] if 'low' in split_k else [split_k.replace('high','low')]
        assert len(unrel_keys) == 1
        rel_vals = [float(variables[all_var][w_two]) for key in unrel_keys for w_two in localizer_words[key]]
        rel_avg = numpy.average(rel_vals)
        dist = -abs(rel_avg-float(variables[all_var][w]))
        distances[w].append(dist)
        ### also trying to match more fundamental variables
                     'log10_word_frequency_sdewac',
                     'old20_score',
                     #'word_average_bigram_frequency',
                     #'word_average_trigram_frequency',
        '''
        ### we want other variables to be matched
        rel_keys = [k for k in localizer_words.keys() if k[-1]==split_k[-1]]
        for rel, more_var in enumerate([
                         'old20_score',
                         'log10_word_frequency_sdewac',
                         'word_length',
                         #'predicted_concreteness',
                         #'predicted_visual',
                         ]):
            rel_vals = [float(variables[more_var][w_two]) for key in rel_keys for w_two in localizer_words[key]]
            rel_avg = numpy.average(rel_vals)
            #dist = (1/(rel+1))*abs(rel_avg-float(variables[more_var][w]))
            dist = (1.5/(rel+1))*abs(rel_avg-float(variables[more_var][w]))
            distances[w].append(dist)
distances = {w : numpy.average(v) for w, v in distances.items()}

best_good = {label : {w : distances[w] for w in v} for label, v in localizer_words.items()}
best_good = {label : [w[0] for w in sorted(v.items(), key=lambda item : item[1])] for label, v in best_good.items()}
for amount_stim in [48, 1000]:
    selected_words = {k : v[:amount_stim] for k, v in best_good.items()}
    if amount_stim != 1000:
        for v in selected_words.values():
            assert len(v) == amount_stim

    ### plotting violinplots
    violin_folder = os.path.join('violins', 'localizer', str(amount_stim))
    os.makedirs(violin_folder, exist_ok=True)
    xs = [val for val in best_good.keys()]
    ### testing p-vals
    ps = list()
    cases = list()
    combs = list(itertools.combinations(xs, r=2))
    for k in relevant_keys:
        vals = {xs[_] : [float(variables[k][w]) for w in selected_words[xs[_]]] for _ in range(len(xs))}
        for c in combs:
            p = scipy.stats.ttest_ind(vals[c[0]], vals[c[1]]).pvalue
            ps.append(p)
            cases.append([k, c[0], c[1]])
        file_name = os.path.join(violin_folder, 'localizer_{}_{}.jpg'.format(str(amount_stim), k))
        fig, ax = pyplot.subplots(constrained_layout=True)
        for _ in range(len(xs)):
            ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in best_good[xs[_]][:amount_stim*2]], showmeans=True)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([x.replace('_', '_') for x in xs])
        ax.set_title('{} distributions for localizer words'.format(k))
        pyplot.savefig(file_name)
        pyplot.clf()
        pyplot.close()

    corrected_ps = mne.stats.fdr_correction(ps)[1]
    #for case, p in zip(cases, ps):
    for case, p in zip(cases, corrected_ps):
        if p<=0.05:
            #print([case, p])
            pass
    #print(k)
    ### corrected p-values
    corr_ps = dict()
    for k in relevant_keys:
        low_sound = [float(variables[k][w])  for _ in xs for w in selected_words[_] if 'lowS' in _]
        hi_sound = [float(variables[k][w]) for _ in xs for w in selected_words[_] if 'highS' in _]
        sound_comp = scipy.stats.ttest_ind(low_sound, hi_sound)
        low_action = [float(variables[k][w]) for _ in xs for w in selected_words[_] if 'lowA' in _]
        hi_action = [float(variables[k][w]) for _ in xs for w in selected_words[_] if 'highA' in _]
        action_comp = scipy.stats.ttest_ind(low_action, hi_action)
        corr_ps[k] = [sound_comp.pvalue, action_comp.pvalue]
    sound_corr = mne.stats.fdr_correction([corr_ps[k][0] for k in relevant_keys])[1]
    action_corr = mne.stats.fdr_correction([corr_ps[k][1] for k in relevant_keys])[1]
    for k_i, k in enumerate(relevant_keys):
        corr_ps[k] = [sound_corr[k_i], action_corr[k_i]]

    ### writing to files the pairwise tests
    res_f = os.path.join('txt_results', 'localizer', str(amount_stim))
    os.makedirs(res_f, exist_ok=True)
    with open(os.path.join(res_f, 'pairwise_comparisons_localizer_{}.tsv'.format(str(amount_stim))), 'w') as o:
        o.write('variable\t')
        o.write('low_sound_avg_zscore\tlow_sound_std\t')
        o.write('high_sound_avg_zscore\thigh_sound_std\t')
        o.write('sound_T\tsound_p_raw\tsound_p_corrected\t')
        o.write('low_action_avg_zscore\tlow_action_std\t')
        o.write('high_action_avg_zscore\thigh_action_std\t')
        o.write('action_T\taction_p_raw\taction_p_corrected\n')
        for k in relevant_keys:
            o.write('{}\t'.format(k))
            ### sound
            low_sound = [float(variables[k][w])  for _ in xs for w in selected_words[_] if 'lowS' in _]
            o.write('{}\t{}\t'.format(round(numpy.average(low_sound), 4),round(numpy.std(low_sound), 4)))
            hi_sound = [float(variables[k][w]) for _ in xs for w in selected_words[_] if 'highS' in _]
            o.write('{}\t{}\t'.format(round(numpy.average(hi_sound), 4),round(numpy.std(hi_sound), 4)))
            stat_comp = scipy.stats.ttest_ind(low_sound, hi_sound)
            o.write('{}\t{}\t'.format(round(stat_comp.statistic, 4),round(stat_comp.pvalue, 5)))
            o.write('{}\t'.format(round(corr_ps[k][0], 5)))
            ### action
            low_action = [float(variables[k][w]) for _ in xs for w in selected_words[_] if 'lowA' in _]
            o.write('{}\t{}\t'.format(round(numpy.average(low_action), 4),round(numpy.std(low_action), 4)))
            hi_action = [float(variables[k][w]) for _ in xs for w in selected_words[_] if 'highA' in _]
            o.write('{}\t{}\t'.format(round(numpy.average(hi_action), 4),round(numpy.std(hi_action), 4)))
            stat_comp = scipy.stats.ttest_ind(low_action, hi_action)
            o.write('{}\t{}\t'.format(round(stat_comp.statistic, 4),round(stat_comp.pvalue, 5)))
            o.write('{}\n'.format(round(corr_ps[k][1], 5)))

    with open(os.path.join(res_f, 'localizer_words_{}.tsv'.format(str(amount_stim))), 'w') as o:
        o.write('word\tcategory\n')
        for cat, ws in selected_words.items():
            for w in ws:
                o.write('{}\t{}\n'.format(w, cat))
