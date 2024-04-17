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
with open('final_exclusions.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        if len(line) == 3:
            if line[1].strip() in ['bad', 'mid']:
                try:
                    corrections[line[2].strip()].append(line[0].strip())
                except KeyError:
                    corrections[line[2].strip()] = [line[0].strip()]

reduced_good = dict()
with open('all_selected_words.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if len(line) == 2:
            word = line[0].strip()
            cat = line[1].strip()
            if word in corrections[cat]:
                print(word)
                continue
            try:
                reduced_good[cat].append(word)
            except KeyError:
                reduced_good[cat] = [word]

for amount_stim in [
                    36, 
                    #42, 
                    #48
                    ]:
    for mode, good in (
                 #('original_exp', old_good),
                 #('good_only', new_good),
                 ('good_manual', reduced_good),
                 #('good_mid', new_mid_good),
                 ):
        print('{}\n\n'.format(mode))
        ### testing p-vals
        ps = list()
        cases = list()

        ### plotting distributions
        print('good items')
        print([(k, len(v)) for k, v in reduced_good.items()])

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
