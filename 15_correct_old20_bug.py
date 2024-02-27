import mne
import numpy
import os
import random
import scipy

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
          #'predicted_dominance',
          #'predicted_arousal',
          ]
relevant_keys = [h for h in variables.keys() if 'en_' not in h and 'raw_' not in h and h not in remove and 'proto' not in h]

mode = 'good_only'
for amount_stim in [36, 42, 48]:
    res_f = os.path.join('txt_results', mode, str(amount_stim))
    os.makedirs(res_f, exist_ok=True)
    ### reading files
    selected_words = dict()
    with open(os.path.join(res_f, 'main_experiment_words_{}_{}.tsv'.format(mode, str(amount_stim)))) as o:
        for l_i, l in enumerate(o):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            if line[1] not in selected_words.keys():
                selected_words[line[1]] = [line[0]]
            else:
                selected_words[line[1]].append(line[0])
            xs = [cat for cat in selected_words.keys()]
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
### localizer
amount_stim = 48
res_f = os.path.join('txt_results', 'localizer', str(amount_stim))
os.makedirs(res_f, exist_ok=True)
### reading files
selected_words = dict()
with open(os.path.join(res_f, 'localizer_words_{}.tsv'.format(str(amount_stim)))) as o:
    for l_i, l in enumerate(o):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        if line[1] not in selected_words.keys():
            selected_words[line[1]] = [line[0]]
        else:
            selected_words[line[1]].append(line[0])
        xs = [cat for cat in selected_words.keys()]
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
