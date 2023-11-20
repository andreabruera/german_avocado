import numpy
import os
import pickle

with open(os.path.join('pickles', 'fourtets_0.05.pkl'), 'rb') as i:
    combs = pickle.load(i)

import pdb; pdb.set_trace()
combs = {c[0] : c[1] for c_i, c in enumerate(combs.items()) if c_1<100000}

### means
means = {k : numpy.average(v) for k, v in combs.items()}
sorted_means = sorted(means.items(), key=lambda item : item[1], reverse=True)
ranking_means = {k[0] : k_i for k_i, k in enumerate(sorted_means.items())}

### stds
stds = {k : numpy.std(v) for k, v in combs.items()}
sorted_stds = sorted(stds.items(), key=lambda item : item[1])
ranking_stds = {k[0] : k_i for k_i, k in enumerate(sorted_stds.items())}

mixed_ranks = dict()
for k, rank_mean in ranking_means.items():
    rank_std = ranking_stds[k]
    mixed_ranks[k] = rank_mean + rank_std

ranked_combs = [w[0] for w in sorted(mixed_ranks.items(), key=lambda item : item[1])]
with open('sorted_fourtets.tsv', 'w') as o:
    o.write('auditory_top\tauditory_bottom\taction_top\taction_bottom\n')
    for r_i, r in enumerate(ranked_combs):
        if r_i < 1000:
            for w in r:
                o.write('{}\t'.format(w))
            o.write('\n')
