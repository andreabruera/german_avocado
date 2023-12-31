import argparse
import multiprocessing
import os
import pickle

from tqdm import tqdm

def read_dewac(file_path):
    with open(file_path) as i:
        sentence = {
                    'word' : list(), 
                    'lemma' : list(),
                    }
        for l in i:
            line = l.strip().split('\t')
            if line[0][:5] in ['<sent', '<erro', '<sour', '<year']: 
                continue
            elif line[0][:3] == '<s>':
                continue
            elif line[0][:6] == '</sent':
                continue
            elif line[0][:4] == '</s>':
                yield sentence
                sentence = {
                    'word' : list(), 
                    'lemma' : list(),
                    }
            if len(line) < 2:
                continue
            else:
                if '$' in line[1]:
                    continue
                else:
                    sentence['word'].append(line[0])
                    sentence['lemma'].append(line[2])

def counter(file_path):
    word_bi_counter = dict()
    word_tri_counter = dict()
    with tqdm() as counter:
        for sentence in read_dewac(file_path):
            for word, lemma in zip(
                                    sentence['word'], 
                                    sentence['lemma'], 
                                    ):
                ### words
                for i, start_let in enumerate(word):
                    ### bigrams
                    if i<len(word)-1:
                        try:
                            word_bi_counter[word[i:i+2]] += 1
                        except KeyError:
                            word_bi_counter[word[i:i+2]] = 1
                    ### trigrams
                    if i<len(word)-2:
                        try:
                            word_tri_counter[word[i:i+3]] += 1
                        except KeyError:
                            word_tri_counter[word[i:i+3]] = 1
                counter.update(1)
    return word_bi_counter, word_tri_counter

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--dewac_path', 
                    required=True,
                    help='path to the folder containing '
                    'the files for the pUkWac dataset'
                    )
args = parser.parse_args()

try:
    assert os.path.exists(args.dewac_path)
except AssertionError:
    raise RuntimeError('The path provided for deWac does not exist!')
paths = [os.path.join(args.dewac_path, f) for f in os.listdir(args.dewac_path)]
try:
    assert len(paths) == 441
except AssertionError:
    raise RuntimeError('(split) deWac is composed by 441 files, but '
                       'the provided folder contains more/less'
                       )

pkls = 'pickles'
os.makedirs(pkls, exist_ok=True)

word_bi_freqs_file = os.path.join(pkls, 'sdewac_word_bi_freqs.pkl')
word_tri_freqs_file = os.path.join(pkls, 'sdewac_word_tri_freqs.pkl')

### Running
with multiprocessing.Pool(processes=int(os.cpu_count()/2)) as pool:
   results = pool.map(counter, paths)
   pool.terminate()
   pool.join()

### Reorganizing results
word_bi_freqs = dict()
word_tri_freqs = dict()
for freq_dict in results:
    ### bi
    for k, v in freq_dict[0].items():
        try:
            word_bi_freqs[k] += v
        except KeyError:
            word_bi_freqs[k] = v
    ### tri
    for k, v in freq_dict[1].items():
        try:
            word_tri_freqs[k] += v
        except KeyError:
            word_tri_freqs[k] = v

with open(word_bi_freqs_file, 'wb') as o:
    pickle.dump(word_bi_freqs, o)
with open(word_tri_freqs_file, 'wb') as o:
    pickle.dump(word_tri_freqs, o)
