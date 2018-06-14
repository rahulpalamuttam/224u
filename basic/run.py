import os
import json

def read_data(path = "data/squad/", data_type="dev", data_filter=None, ref=True):
    data_path = path + "data_{}.json".format(data_type)
    shared_path = path + "shared_{}.json".format(data_type)

    data_path = os.path.join(data_path)
    shared_path = os.path.join(shared_path)

    with open(data_path, 'r') as fh:
        data = json.load(fh)
    with open(shared_path, 'r') as fh:
        shared = json.load(fh)

    num_examples = len(next(iter(data.values())))
    
        
    if data_filter is None:
        valid_idxs = range(num_examples)
    else:
        mask = []
        keys = data.keys()
        values = data.values()
        for vals in zip(*values):
            each = {key: val for key, val in zip(keys, vals)}
            mask.append(data_filter(each, shared))
        valid_idxs = [idx for idx in range(len(mask)) if mask[idx]]

    print("Loaded {}/{} examples from {}".format(len(valid_idxs), num_examples, data_type))

    shared_path = "out"
    new_shared = json.load(open(shared_path, 'r'))
    for key, val in new_shared.items():
        shared[key] = val

        
    # create new word2idx and word2vec
    word2vec_dict = shared['lower_word2vec']
    new_word2idx_dict = {word: idx for idx, word in enumerate(word for word in word2vec_dict.keys() if word not in shared['word2idx'])}
    shared['new_word2idx'] = new_word2idx_dict
    offset = len(shared['word2idx'])
    word2vec_dict = shared['lower_word2vec'] if config.lower_word else shared['word2vec']
    new_word2idx_dict = shared['new_word2idx']
    idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
    print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
    new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
    shared['new_emb_mat'] = new_emb_mat
    
    data_set = DataSet(data, data_type, shared=shared, valid_idxs=valid_idxs)
    return data_set

read_data()
