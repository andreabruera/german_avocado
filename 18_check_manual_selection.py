import os

mode = 'good_manual'

for amount_stim in [
                    36, 
                    #42, 48
                    ]:
    res_f = os.path.join('two_lists_manual', mode, str(amount_stim))
    words = {'one' : dict(), 'two' : dict()}
    for idx in words.keys():
        with open(os.path.join(res_f, '{}_main_experiment_words_{}_{}.tsv'.format(idx, mode, str(amount_stim)))) as o:
            for l_i, l in enumerate(o):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                try:
                    words[idx][line[1]].append(line[0])
                except KeyError:
                    words[idx][line[1]] = [line[0]]
        keys = [k for k in words[idx].keys()]
    for k in keys:
        assert len(set(words['one'][k])) == amount_stim
        assert len(set(words['two'][k])) == amount_stim
        assert len(set(words['one'][k]) & set(words['two'][k])) == 0
        p_val = scipy.stats.
