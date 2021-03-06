def get_config_penn():
    config = {}
    ####### network param #############
    config['seq_len'] = 80
    config['n_hids'] = 200
    config['n_emb_morph'] = 200
    config['n_shids'] = 200                     # structure hidden dimension
    config['n_structs'] = 4
    config['n_emb_lstm'] = 200
    config['n_emb_struct'] = 200
    config['n_emb_share'] = 0
    config['batch_size'] = 100
    config['batch_size_valid'] = 100
    config['vocab_size'] = 10001

    ####### devil in detail ##############
    config['use_noise'] = 1                     # use noise = 1 mean use dropout layer ?
    config['dropout'] = 0.5
    config['hard_clipping'] = -1.               # hard clip grad when > 0, hard clip take effect before soft clip
    config['soft_clipping_epoch'] = 30          # soft clipping decrease from begin to end in given epoches
    config['soft_clipping_begin'] = 30.
    config['soft_clipping_end'] = 1            # theoretically, only this effect result in all 3 soft_clipping params
    config['skip_nan_grad'] = 1                 # 1 means skip
    config['shrink_scale_after_skip_nan_grad'] = 1.0

    ####### loop behave ##############
    config['is_shuffle'] = True
    config['drop_last_batch_if_small'] = True
    config['check_freq'] = 5100
    config['check_low_freq'] = 50000
    config['check_low_freq_count'] = 1500000

    ####### path ##############
    datadir = '../data/en_data/'
    config['train_file'] = datadir + 'train'
    config['valid_file'] = datadir + 'valid'
    config['test_file'] = datadir + 'test'

    ####### dict ##############
    config['train_dic'] = datadir + 'train_dic.pkl'
    config['unk_id'] = 0                        # The value is fixed, dont change it.
    config['bos_token'] = '<s>'
    config['eos_token'] = '</s>'
    #config['eos_token'] = None
    config['unk_token'] = '<unk>'

    return config


def get_config_morph():
    config = {}
    ####### network param #############
    config['seq_len'] = 40
    config['seq_morph_len'] = 50
    config['n_hids'] = 200
    config['n_emb_morph'] = 200
    config['n_emb_lstm'] = 200
    config['batch_size'] = 100
    config['batch_size_valid'] = 100
    config['vocab_size'] = 10001
    config['morph_size'] = 10001

    ####### devil in detail ##############
    config['use_noise'] = 1                     # use noise = 1 mean use dropout layer ?
    config['dropout'] = 0.5
    config['hard_clipping'] = -1.               # hard clip grad when > 0, hard clip take effect before soft clip
    config['soft_clipping_epoch'] = 30          # soft clipping decrease from begin to end in given epoches
    config['soft_clipping_begin'] = 30.
    config['soft_clipping_end'] = 1            # theoretically, only this effect result in all 3 soft_clipping params
    config['skip_nan_grad'] = 1                 # 1 means skip
    config['shrink_scale_after_skip_nan_grad'] = 1.0

    ####### loop behave ##############
    config['is_shuffle'] = True
    config['drop_last_batch_if_small'] = True
    config['check_freq'] = 5100
    config['check_low_freq'] = 50000
    config['check_low_freq_count'] = 1500000

    ####### path ##############
    datadir = '../data/uy_data2/'
    config['train_file'] = datadir + 'train_word'
    config['train_morph_file'] = datadir + 'train_morph'
    config['train_rel_file'] = datadir + 'train_rel'
    config['valid_file'] = datadir + 'valid_word'
    config['valid_morph_file'] = datadir + 'valid_morph'
    config['valid_rel_file'] = datadir + 'valid_rel'
    config['test_file'] = datadir + 'test_word'
    config['test_morph_file'] = datadir + 'test_morph'
    config['test_rel_file'] = datadir + 'test_rel'

    ####### dict ##############
    config['train_dic'] = datadir + 'train_dic.pkl'
    config['train_morph_dic'] = datadir + 'train_morph_dic.pkl'
    config['unk_id'] = 0                        # The value is fixed, dont change it.
    config['bos_token'] = '<s>'
    config['eos_token'] = '</s>'
    #config['eos_token'] = None
    config['unk_token'] = '<unk>'

    return config




def get_config_morph2():
    config = {}
    ####### network param #############
    config['seq_len'] = 40
    config['seq_morph_len'] = 50
    config['n_hids'] = 200
    config['n_emb_morph'] = 200
    config['n_emb_lstm'] = 200
    config['batch_size'] = 100
    config['batch_size_valid'] = 100
    config['vocab_size'] = 10001
    config['morph_size'] = 10001

    ####### devil in detail ##############
    config['use_noise'] = 1                     # use noise = 1 mean use dropout layer ?
    config['dropout'] = 0.5
    config['hard_clipping'] = -1.               # hard clip grad when > 0, hard clip take effect before soft clip
    config['soft_clipping_epoch'] = 30          # soft clipping decrease from begin to end in given epoches
    config['soft_clipping_begin'] = 30.
    config['soft_clipping_end'] = 1            # theoretically, only this effect result in all 3 soft_clipping params
    config['skip_nan_grad'] = 1                 # 1 means skip
    config['shrink_scale_after_skip_nan_grad'] = 1.0

    ####### loop behave ##############
    config['is_shuffle'] = True
    config['drop_last_batch_if_small'] = True
    config['check_freq'] = 5100
    config['check_low_freq'] = 50000
    config['check_low_freq_count'] = 1500000

    ####### path ##############
    datadir = '../data/cs/'
    config['train_file'] = datadir + 'train_word'
    config['train_morph_file'] = datadir + 'train_morph'
    config['train_rel_file'] = datadir + 'train_rel'
    config['valid_file'] = datadir + 'dev_word'
    config['valid_morph_file'] = datadir + 'dev_morph'
    config['valid_rel_file'] = datadir + 'dev_rel'
    config['test_file'] = datadir + 'test_word'
    config['test_morph_file'] = datadir + 'test_morph'
    config['test_rel_file'] = datadir + 'test_rel'

    ####### dict ##############
    config['train_dic'] = datadir + 'train_dic.pkl'
    config['train_morph_dic'] = datadir + 'train_morph_dic.pkl'
    config['unk_id'] = 0                        # The value is fixed, dont change it.
    config['bos_token'] = '<s>'
    config['eos_token'] = '</s>'
    #config['eos_token'] = None
    config['unk_token'] = '<unk>'

    return config

