''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants
import numpy as np

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count, voc_size):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}
    nb_special_words = len(word2idx)
    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1


    if voc_size <= 0:
        ignored_word_count = 0
        for word, count in word_count.items():
            if word not in word2idx:
                if count > min_word_count:
                    word2idx[word] = len(word2idx)
                else:
                    ignored_word_count += 1
        print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
                'each with minimum occurrence = {}'.format(min_word_count))
    else:
        #####################################################################
        voc_size = min(voc_size, len(word_count)+len(word2idx))

        vec_voc = []
        vec_count = []
        for w in word_count:
            if w not in word2idx:
                vec_voc += [w]
                vec_count += [word_count[w]]

        vec_voc = np.array(vec_voc)
        vec_count = np.array(vec_count)
        index_sorted = np.argsort(vec_count)
        vec_voc = (vec_voc[index_sorted])[::-1]
        #vec_count = (vec_count[index_sorted])[::-1]

        for i in range(voc_size-len(word2idx)):
            word = vec_voc[i]
            word2idx[word] = len(word2idx)

        ignored_word_count = len(vec_voc) - (voc_size-nb_special_words)
        print('[Warning] voc_size is greater then 0, min_word_count will be ignored')
        #####################################################################


    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    '''Word mapping to idx'''
    return [[word2idx[w] if w in word2idx else Constants.UNK for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True)
    parser.add_argument('-train_tgt', required=True)
    parser.add_argument('-train_ctx', required=True)
    parser.add_argument('-valid_src', required=True)
    parser.add_argument('-valid_tgt', required=True)
    parser.add_argument('-valid_ctx', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-voc_size', type=int, default=-1)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_src_word_insts = read_instances_from_file(
        opt.train_src, opt.max_word_seq_len, opt.keep_case)
    train_tgt_word_insts = read_instances_from_file(
        opt.train_tgt, opt.max_word_seq_len, opt.keep_case)
    train_ctx_word_insts = read_instances_from_file(
        opt.train_ctx, opt.max_word_seq_len, opt.keep_case)

    if not (len(train_src_word_insts) == len(train_tgt_word_insts) == len(train_ctx_word_insts)):
        print('[Warning] The training instance count is not equal.')
        min_inst_count = min(len(train_src_word_insts), len(train_tgt_word_insts), len(train_ctx_word_insts))
        train_src_word_insts = train_src_word_insts[:min_inst_count]
        train_tgt_word_insts = train_tgt_word_insts[:min_inst_count]
        train_ctx_word_insts = train_ctx_word_insts[:min_inst_count]

    #- Remove empty instances
    train_src_word_insts, train_tgt_word_insts, train_ctx_word_insts = list(zip(*[
        (s, t, c) for s, t, c in zip(train_src_word_insts, train_tgt_word_insts, train_ctx_word_insts) if s and t]))

    # Validation set
    valid_src_word_insts = read_instances_from_file(
        opt.valid_src, opt.max_word_seq_len, opt.keep_case)
    valid_tgt_word_insts = read_instances_from_file(
        opt.valid_tgt, opt.max_word_seq_len, opt.keep_case)
    valid_ctx_word_insts = read_instances_from_file(
        opt.valid_ctx, opt.max_word_seq_len, opt.keep_case)

    if not (len(valid_src_word_insts) == len(valid_tgt_word_insts) == len(valid_ctx_word_insts)) :
        print('[Warning] The validation instance count is not equal.')
        min_inst_count = min(len(valid_src_word_insts), len(valid_tgt_word_insts), len(valid_ctx_word_insts))
        valid_src_word_insts = valid_src_word_insts[:min_inst_count]
        valid_tgt_word_insts = valid_tgt_word_insts[:min_inst_count]
        valid_ctx_word_insts = valid_ctx_word_insts[:min_inst_count]

    #- Remove empty instances
    valid_src_word_insts, valid_tgt_word_insts, valid_ctx_word_insts = list(zip(*[
        (s, t, c) for s, t, c in zip(valid_src_word_insts, valid_tgt_word_insts, valid_ctx_word_insts) if s and t]))

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        src_word2idx = predefined_data['dict']['src']
        tgt_word2idx = predefined_data['dict']['tgt']
    else:
        if opt.share_vocab:
            print('[Info] Build shared vocabulary for source and target.')
            word2idx = build_vocab_idx(
                train_src_word_insts + train_tgt_word_insts, opt.min_word_count, opt.voc_size)
            src_word2idx = tgt_word2idx = word2idx
        else:
            print('[Info] Build vocabulary for source.')
            src_word2idx = build_vocab_idx(train_src_word_insts, opt.min_word_count, opt.voc_size)
            print('[Info] Build vocabulary for target.')
            tgt_word2idx = build_vocab_idx(train_tgt_word_insts, opt.min_word_count, opt.voc_size)

    # word to index
    print('[Info] Convert source word instances into sequences of word index.')
    train_src_insts = convert_instance_to_idx_seq(train_src_word_insts, src_word2idx)
    valid_src_insts = convert_instance_to_idx_seq(valid_src_word_insts, src_word2idx)

    print('[Info] Convert target word instances into sequences of word index.')
    train_tgt_insts = convert_instance_to_idx_seq(train_tgt_word_insts, tgt_word2idx)
    valid_tgt_insts = convert_instance_to_idx_seq(valid_tgt_word_insts, tgt_word2idx)

    print('[Info] Convert context word instances into sequences of word index.')
    train_ctx_insts = convert_instance_to_idx_seq(train_ctx_word_insts, src_word2idx)
    valid_ctx_insts = convert_instance_to_idx_seq(valid_ctx_word_insts, src_word2idx)

    data = {
        'settings': opt,
        'dict': {
            'src': src_word2idx,
            'tgt': tgt_word2idx},
        'train': {
            'src': train_src_insts,
            'tgt': train_tgt_insts,
            'ctx': train_ctx_insts},
        'valid': {
            'src': valid_src_insts,
            'tgt': valid_tgt_insts,
            'ctx': valid_ctx_insts}}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
