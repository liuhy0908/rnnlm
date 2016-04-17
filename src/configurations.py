def get_config_penn():
    config = {}
    ####### network param #############
    config['seq_len'] = 80
    config['n_hids'] = 200
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
    ########### loop behave ##############
    #config['check_freq'] = 51
    #config['check_low_freq'] = 500
    #config['check_low_freq_count'] = 1500

    ####### path ##############
    datadir = '../data/'
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

