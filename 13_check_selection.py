import os

mode = 'good_only'

for amount_stim in [36, 42, 48]:
    res_f = os.path.join('two_lists', mode, str(amount_stim))
    words = {'one' : dict(), 'two' : dict()}
    keys = set()
    for idx in words.keys():
        with open(os.path.join(res_f, '{}_main_experiment_words_{}_{}.tsv'.format(idx, mode, str(amount_stim)))) as o:
            for l_i, l in enumerate(o):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                if line[1] not in words[idx].keys():
                    words[idx][line[1]] = set(line[0])
                    keys.add(line[1])
                else:
                    words[idx][line[1]].add(line[0])
    for k in keys:
        print(len(words['one'][k]))
        print(len(words['two'][k]))
        assert len(words['one'][k]) == amount_stim
        assert len(words['two'][k]) == amount_stim
        assert len(words['one'][k] | words['two'][k]) == 0
