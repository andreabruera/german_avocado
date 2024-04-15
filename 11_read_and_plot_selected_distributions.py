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
    if label == 'highA_lowS':
        if float(variables['predicted_concreteness'][word]) > 2.:
            marker = True
    ### this is where we can cut the most...
    if label == 'lowA_lowS':
        #if float(variables['predicted_concreteness'][word]) < 1.1:
        #    marker = True
        if float(variables['old20_score'][word]) <= 2.:
            marker = True
    #if label == 'lowA_highS':
    #    if float(variables['old20_score'][word]) >= 4.:
    #        marker = True
    if label == 'highA_highS':
        if float(variables['old20_score'][word]) >= 5.:
            marker = True
    '''
        if float(variables['word_average_trigram_frequency'][word]) > 10500000:
            marker = True
        if float(variables['log10_word_frequency_sdewac'][word]) > 3.9:
                marker = True
    '''
    return marker

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
          #'predicted_visual',
          #'predicted_olfactory',
          #'predicted_gustatory',
          #'predicted_hearing',
          ]
relevant_keys = [h for h in variables.keys() if 'en_' not in h and 'raw_' not in h and h not in remove and 'proto' not in h]

### reading latest corrections
corrections = dict()
with open('phil_correction_21_02.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        if len(line) > 2:
            if line[2].strip() in ['bad', 'mid']:
                corrections[line[0].strip()] = line[2].strip()

### reading words from previous experiments
old_file = os.path.join('output', 'phil_original_annotated_clean.tsv')
old_good = dict()
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
        if label not in old_good.keys():
            old_good[label] = set()
            localizer[label] = set()
        marker = False
        ### words from the original experiment that are missing...
        if line[word] not in full_words:
            print(line[word])
            continue
        ### pruning distributions
        if skip_words(line[word], label) == True:
            continue
        ### checking evaluation is high enough, else localizer
        if eval_val < 0.9:
            if eval_val > 0.5:
                localizer[label].add(line[word])
        else:
            old_good[label].add(line[word])
print('old items')
print([(k, len(v)) for k, v in old_good.items()])
print('old localizers')
print([(k, len(v)) for k, v in localizer.items()])

old_good = {l : v for l, v in old_good.items()}
### reading selected nouns
new_good = {l : v for l, v in old_good.items()}
new_mid_good = {l : v for l, v in old_good.items()}

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
            if skip_words(line[0], label) == True:
                continue
            ### dividing into bad/mid/good
            if line[1] in ['bad']:
                continue
            elif line[1] in ['mid']:
                new_mid_good[label].add(line[0])
            elif line[1] in ['action', 'sound']:
                localizer[label].add(line[0])
            else:
                ### good
                #print(line[0])
                new_mid_good[label].add(line[0])
                new_good[label].add(line[0])
print('localizer items')
print([(k, len(v)) for k, v in localizer.items()])

old_good = {l : set([w for w in v if w not in corrections.keys()]) for l, v in old_good.items()}
### reading selected nouns
new_good = {l : set([w for w in v if w not in corrections.keys()]) for l, v in new_good.items()}
new_mid_good = {l : set([w for w in v if w not in corrections.keys() or corrections[w]=='mid']) for l, v in new_mid_good.items()}

### reading latest corrections
corrections = list()
with open('phil_correction_15_04.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        if len(line) > 2:
            if line[2].strip() in ['bad', 'mid']:
                #print(line[0])
                corrections.append(line[0].strip())
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
                corrections.append(line[0].strip())

res_f = os.path.join('txt_results', 'localizer', '48')
with open(os.path.join(res_f, 'localizer_words_48.tsv')) as o:
    for l_i, l in enumerate(o):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        corrections.append(line[0].strip())
reduced_good = {k : [w for w in v if w not in corrections] for k, v in new_good.items()}
print([(k, len(v)) for k, v in reduced_good.items()])
for k in reduced_good.keys():
    new_l = list()
    for w in reduced_good[k]:
        if 'lowS' in k:
            if float(variables['predicted_auditory'][w]) > .5:
                continue
        if 'highS' in k:
            if float(variables['predicted_auditory'][w]) < -.5:
                continue
        if 'lowA' in k:
            if float(variables['predicted_hand'][w]) > .3:
                continue
        if 'highA' in k:
            if float(variables['predicted_hand'][w]) < -.3:
                continue
        new_l.append(w)
    reduced_good[k] = new_l
print([(k, len(v)) for k, v in reduced_good.items()])

for amount_stim in [
                    36, 
                    #42, 
                    #48
                    ]:
    for mode, good in (
                 #('original_exp', old_good),
                 #('good_only', new_good),
                 ('good_only', reduced_good),
                 #('good_mid', new_mid_good),
                 ):
        print('{}\n\n'.format(mode))
        ### testing p-vals
        ps = list()
        cases = list()

        ### plotting distributions
        print('good items')
        print([(k, len(v)) for k, v in good.items()])
        print('localizer items')
        print([(k, len(v)) for k, v in localizer.items()])

        ### plotting violinplots
        violin_folder = os.path.join('violins', mode, str(amount_stim))
        os.makedirs(violin_folder, exist_ok=True)
        xs = [val for val in good.keys()]
        for k in relevant_keys:
            #print(k)
            xs = [val for val in good.keys()]
            combs = list(itertools.combinations(xs, r=2))
            vals = {xs[_] : [float(variables[k][w]) for w in good[xs[_]]] for _ in range(len(xs))}
            for c in combs:
                p = scipy.stats.ttest_ind(vals[c[0]], vals[c[1]]).pvalue
                ps.append(p)
                cases.append([k, c[0], c[1]])
            file_name = os.path.join(violin_folder, 'all_{}_{}_{}.jpg'.format(k, mode, str(amount_stim)))
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

        distances = dict()
        for good_k, v in good.items():
            split_k = good_k.split('_')
            ### every word
            for w in v:
                distances[w] = list()
                ### the thing we really care about are hand and audition
                for var_i, var in enumerate([
                                             'predicted_hand', 
                                             'predicted_auditory',
                                             ]):
                    ### promoting similarity
                    rel_keys = [k for k in good.keys() if split_k[var_i] in k and k!=good_k]
                    #assert len(rel_keys) == 2
                    assert len(rel_keys) == 1
                    all_vars = [var, 'predicted_concreteness']
                    #all_vars = [var]
                    for all_var in all_vars:
                        rel_vals = [float(variables[all_var][w_two]) for key in rel_keys for w_two in good[key]]
                        rel_avg = numpy.average(rel_vals)
                        dist = abs(rel_avg-float(variables[all_var][w]))
                        distances[w].append(dist)
                    '''
                    ### promoting dissimilarity
                    unrel_keys = [k for k in good.keys() if split_k[var_i] not in k]
                    assert len(unrel_keys) == 2
                    rel_vals = [float(variables[all_var][w_two]) for key in unrel_keys for  w_two in good[key]]
                    rel_avg = numpy.average(rel_vals)
                    dist = -abs(rel_avg-float(variables[all_var][w]))
                    distances[w].append(dist)
                    '''
                    ### also trying to match more fundamental variables
                    '''
                                 'log10_word_frequency_sdewac',
                                 'old20_score',
                                 #'word_average_bigram_frequency',
                                 #'word_average_trigram_frequency',
                    '''
                    for rel, more_var in enumerate([
                                     'old20_score',
                                     'log10_word_frequency_sdewac',
                                     #'word_length',
                                     #'predicted_concreteness',
                                     #'predicted_visual',
                                     ]):
                        rel_vals = [float(variables[more_var][w_two]) for key in rel_keys for w_two in good[key]]
                        rel_avg = numpy.average(rel_vals)
                        #dist = (1/(rel+1))*abs(rel_avg-float(variables[more_var][w]))
                        dist = (1.5/(rel+1))*abs(rel_avg-float(variables[more_var][w]))
                        distances[w].append(dist)
        distances = {w : numpy.average(v) for w, v in distances.items()}

        best_good = {label : {w : distances[w] for w in v} for label, v in good.items()}
        best_good = {label : [w[0] for w in sorted(v.items(), key=lambda item : item[1])] for label, v in best_good.items()}
        if mode == 'original_exp':
            selected_words = {k : v[:amount_stim] for k, v in best_good.items()}
            for v in selected_words.values():
                assert len(v) == amount_stim
        else:
            selected_words = {k : v[:amount_stim*2] for k, v in best_good.items()}
            for v in selected_words.values():
                assert len(v) == amount_stim*2
        ### hard-coding phil's words
        hard_code = [
                     'Pferd', 
                     'Kind', 
                     'Baby', 
                     'Säugling',
                     ]
        hard_code = [w for w in hard_code if w not in selected_words['highA_highS']]
        selected_words['highA_highS'] = hard_code + selected_words['highA_highS']
        selected_words['highA_highS'] = selected_words['highA_highS'][:amount_stim*2]
        ### criterion: average separately for high/low action/sound
        #best_good = {k : random.sample(list(v), k=len(v)) for k, v in good.items()}

        ### plotting violinplots
        violin_folder = os.path.join('violins', 'selected', mode, str(amount_stim))
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
            file_name = os.path.join(violin_folder, '{}_{}_{}.jpg'.format(str(amount_stim), mode, k))
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
        res_f = os.path.join('txt_results', mode, str(amount_stim))
        os.makedirs(res_f, exist_ok=True)
        with open(os.path.join(res_f, 'pairwise_comparisons_main_experiment_{}_{}.tsv'.format(mode, str(amount_stim))), 'w') as o:
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

        with open(os.path.join(res_f, 'main_experiment_words_{}_{}.tsv'.format(mode, str(amount_stim))), 'w') as o:
            o.write('word\tcategory\n')
            for cat, ws in selected_words.items():
                for w in ws:
                    o.write('{}\t{}\n'.format(w, cat))

'''
### localizer now
xs = [val for val in best_good.keys()]
passage_localizer = {_ : set(best_good[_][amount_stim*2:]) | localizer[_] for _ in xs}
all_localizer = {

distances = dict()
for k, v in all_localizer.items():
    split_k = k.split('_')
    ### every word
    for w in v:
        distances[w] = list()
        ### the thing we really care about are hand and audition
        for var_i, var in enumerate(['predicted_hand', 'predicted_auditory']):
            rel_keys = [k for k in all_localizer.keys() if split_k[var_i] in k]
            rel_vals = [float(variables[var][w_two]) for key in rel_keys for  w_two in all_localizer[key]]
            dist = abs(numpy.average(rel_vals)-float(variables[var][w]))
            distances[w].append(dist)
distances = {w : numpy.average(v) for w, v in distances.items()}

best_localizer = {label : {w : distances[w] for w in v} for label, v in all_localizer.items()}
best_localizer = {label : [w[0] for w in sorted(v.items(), key=lambda item : item[1])][:int(amount_stim*0.5)] for label, v in best_localizer.items()}
### criterion: average separately for high/low action/sound
#best_good = {k : random.sample(list(v), k=len(v)) for k, v in good.items()}
for v in best_localizer.values():
    assert len(v) == int(amount_stim*0.5)

### plotting violinplots
violin_folder = os.path.join('violins', 'best_for_localizer')
os.makedirs(violin_folder, exist_ok=True)
for k in relevant_keys:
    print(k)
    file_name = os.path.join(violin_folder, '{}.jpg'.format(k))
    fig, ax = pyplot.subplots(constrained_layout=True)
    for _ in range(len(xs)):
        ax.violinplot(positions=[_], dataset=[float(variables[k][w]) for w in best_localizer[xs[_]]], showmeans=True)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([x.replace('_', '_') for x in xs])
    ax.set_title('{} distributions for selected words'.format(k))
    pyplot.savefig(file_name)
    pyplot.clf()
    pyplot.close()

### writing to files the pairwise tests
with open('pairwise_comparisons_localizer.tsv', 'w') as o:
    o.write('variable\tlow_sound_avg_zscore\thigh_sound_std\tsound_T\tsound_p\t'\
                      'low_action_avg_zscore\thigh_action_std\taction_T\taction_p\n')
    for k in relevant_keys:
        o.write('{}\t'.format(k))
        ### sound
        low_sound = [[float(variables[k][w]) for w in best_localizer[_] for _ in xs if 'lowS' in _]
        o.write('{}\t{}\t'.format(round(numpy.average(low_sound), 4),round(numpy.std(low_sound), 4)))
        hi_sound = [[float(variables[k][w]) for w in best_localizer[_] for _ in xs if 'highS' in _]
        o.write('{}\t{}\t'.format(round(numpy.average(hi_sound), 4),round(numpy.std(hi_sound), 4)))
        stat_comp = scipy.stats.ttest_ind(low_sound, hi_sound)
        o.write('{}\t{}\t'.format(round(stat_comp.statistic, 4),round(stat_comp.pvalue, 5)))
        ### action
        low_action = [[float(variables[k][w]) for w in best_localizer[_] for _ in xs if 'lowA' in _]
        o.write('{}\t{}\t'.format(round(numpy.average(low_action), 4),round(numpy.std(low_action), 4)))
        hi_action = [[float(variables[k][w]) for w in best_localizer[_] for _ in xs if 'highA' in _]
        o.write('{}\t{}\t'.format(round(numpy.average(hi_action), 4),round(numpy.std(hi_action), 4)))
        stat_comp = scipy.stats.ttest_ind(low_action, hi_action)
        o.write('{}\t{}\n'.format(round(stat_comp.statistic, 4),round(stat_comp.pvalue, 5)))

with open('main_experiment_words.tsv', 'w') as o:
    o.write('word\tcategory\n')
    for cat, ws in selected_words.items():
        for w in ws:
            o.write('{}\t{}\n'.format(w, cat))
'''
