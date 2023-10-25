import numpy
import os
import pickle

### reading phil's candidate dataset
relevant_words = list()
with open(os.path.join('output', 'candidate_nouns_min_100.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i==0:
            continue
        relevant_words.append(line[0])

with open(os.path.join('pickles', 'sdewac_lemma_freqs.pkl'), 'rb') as i:
    lemma_freqs = pickle.load(i)
with open(os.path.join('pickles', 'sdewac_word_freqs.pkl'), 'rb') as i:
    word_freqs = pickle.load(i)
with open(os.path.join('pickles', 'sdewac_word_bi_freqs.pkl'), 'rb') as i:
    word_bi_freqs = pickle.load(i)
with open(os.path.join('pickles', 'sdewac_word_tri_freqs.pkl'), 'rb') as i:
    word_tri_freqs = pickle.load(i)

with open(os.path.join('output', 'candidate_nouns_freqs.tsv'), 'w') as o:
    o.write('word\traw_word_frequency_sdewac\tlog10_word_frequency_sdewac\traw_lemma_frequency_sdewac\tlog10_lemma_frequency_sdewac\tword_average_bigram_frequency\tword_average_trigram_frequency\n')
    for w in relevant_words:
        o.write('{}\t'.format(w))
        try:
            o.write('{}\t'.format(word_freqs[w]))
            o.write('{}\t'.format(round(numpy.log10(word_freqs[w]), 6)))
        except KeyError:
            o.write('na\tna\t')
        o.write('{}\t'.format(lemma_freqs[w]))
        o.write('{}\t'.format(round(numpy.log10(lemma_freqs[w]), 6)))
        ### bigram freq
        ### words
        bigram = list()
        trigram = list()
        for i, start_let in enumerate(w):
            ### bigrams
            if i<len(w)-1:
                bi = word_bi_freqs[w[i:i+2]]
                bigram.append(bi)
            ### trigrams
            if i<len(w)-2:
                tri = word_tri_freqs[w[i:i+3]]
                trigram.append(tri)
        o.write('{}\t'.format(round(numpy.average(bigram), 2)))
        o.write('{}\t'.format(round(numpy.average(trigram), 2)))
        o.write('\n')
