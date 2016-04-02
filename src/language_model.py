import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
from model import GRU, LSTM, SLSTM, lookup_table, chunk_layer, dropout_layer, maxout_layer, rnn_pyramid_layer, LogisticRegression
from utils import adadelta, step_clipping
from stream import DStream
import numpy
from datetime import datetime,timedelta                                          
import time
import sys
import logging
import pprint

logger = logging.getLogger(__name__)

class language_model(object):

    def __init__(self, vocab_size, n_emb_lstm, n_emb_struct, n_emb_share, n_hids, n_shids, n_structs, dropout, **kwargs):
        self.n_hids = n_hids
        self.n_shids = n_shids # kwargs.pop('n_shids', n_hids)
        self.n_structs = n_structs
        self.n_emb_lstm = n_emb_lstm
        self.n_emb_struct = n_emb_struct
        self.n_emb_share = n_emb_share #kwargs.pop('n_emb_share', min(n_emb_lstm, n_emb_struct)/2)
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.params = []
        self.layers = []

    def apply(self, sentence, sentence_mask, use_noise=1):
        n_emb_lstm = self.n_emb_lstm
        n_emb_struct = self.n_emb_struct
        n_emb_share = self.n_emb_share

        src = sentence[:-1]
        src_mask = sentence_mask[:-1]
        tgt = sentence[1:]
        tgt_mask = sentence_mask[1:]

        if False: #(share only part of embedding)
            n_emb_all = n_emb_lstm + n_emb_struct - n_emb_share
            emb_all_range = T.arange(n_emb_all)
            emb_lstm_range = T.arange(n_emb_lstm)
            emb_struct_range = T.arange(n_emb_lstm - n_emb_share, n_emb_all)

            table = lookup_table(n_emb_all, self.vocab_size, name='Wemb')
            state_below = table.apply(src, emb_all_range)
            state_below_lstm = table.apply(src, emb_lstm_range)
            state_below_struct = table.apply(src, emb_struct_range)
            self.layers.append(table)

            rnn = SLSTM(n_emb_lstm, n_emb_struct, n_emb_share, self.n_hids, self.n_shids, self.n_structs)
            #rnn = LSTM(self.n_in, self.n_hids)
            hiddens = rnn.merge_out(state_below, state_below_lstm, state_below_struct, src_mask)
            self.layers.append(rnn)

        elif True: # use rnn_pyramid
            emb_lstm_range = T.arange(n_emb_lstm)
            table = lookup_table(n_emb_lstm, self.vocab_size, name='Wemb')
            state_below = table.apply(src, emb_lstm_range)
            self.layers.append(table)

            if self.dropout < 1.0:
                state_below = dropout_layer(state_below, use_noise, self.dropout)

            rnn = rnn_pyramid_layer(n_emb_lstm, self.n_hids)
            hiddens, cells, structs = rnn.apply(state_below, src_mask)
            self.layers.append(rnn)
            self.structs = structs

        else: # share all embedding
            emb_lstm_range = T.arange(n_emb_lstm)
            table = lookup_table(n_emb_lstm, self.vocab_size, name='Wemb')
            state_below = table.apply(src, emb_lstm_range)
            self.layers.append(table)

            if self.dropout < 1.0:
                state_below = dropout_layer(state_below, use_noise, self.dropout)

            rnn = LSTM(n_emb_lstm, self.n_hids)
            hiddens, cells = rnn.apply(state_below, src_mask)
            #hiddens = rnn.merge_out(state_below, src_mask)
            self.layers.append(rnn)

            if self.dropout < 1.0:
                hiddens = dropout_layer(hiddens, use_noise, self.dropout)

            rnn1 = LSTM(self.n_hids, self.n_hids)
            hiddens, cells = rnn1.apply(hiddens, src_mask)
            #hiddens = rnn.merge_out(state_below, src_mask)
            self.layers.append(rnn1)

            maxout = maxout_layer()
            states = T.concatenate([state_below, hiddens], axis=2)
            hiddens = maxout.apply(states, n_emb_lstm + self.n_hids, self.n_hids, src_mask)
            self.layers.append(maxout)

            #rnng = LSTM(n_emb_lstm, self.n_hids)
            #hiddens, cells = rnn.apply(state_below, src_mask)
            #hiddensg = rnng.merge_out(state_below, src_mask)
            #self.layers.append(rnng)

            if self.dropout < 1.0:
                hiddens = dropout_layer(hiddens, use_noise, self.dropout)

            #chunk = chunk_layer(n_lstm_in + n_lstm_out, n_lstm_out, n_chunk_out, 6)
            n_emb_hid = n_emb_lstm + self.n_hids
            emb_hid = T.concatenate([state_below, hiddens], axis=2)
            #chunk = chunk_layer(self.n_hids, self.n_hids, self.n_hids, self.n_structs)
            #hiddens = chunk.merge_out(hiddens, hiddens, src_mask, merge_how="for_struct",\
            #        state_below_other=state_below, n_other=n_emb_lstm)
            chunk = chunk_layer(n_emb_hid, self.n_hids, self.n_hids, self.n_structs)
            hiddens = chunk.merge_out(emb_hid, hiddens, src_mask, merge_how="for_struct",\
                    state_below_other=None, n_other=0)
            #chunk = chunk_layer(self.n_hids, self.n_hids, self.n_hids, self.n_structs)
            #hiddens = chunk.merge_out(hiddens, hiddensg, src_mask, merge_how="both",\
            #        state_below_other=state_below, n_other=n_emb_lstm)
            self.layers.append(chunk)

        # apply dropout
        if self.dropout < 1.0:
            # dropout is applied to the output of maxout in ghog
            hiddens = dropout_layer(hiddens, use_noise, self.dropout)

        logistic_layer = LogisticRegression(hiddens, self.n_hids, self.vocab_size)
        self.layers.append(logistic_layer)

        self.cost = logistic_layer.cost(tgt, tgt_mask)

        for layer in self.layers:
            self.params.extend(layer.params)

        self.L2 = sum(T.sum(item ** 2) for item in self.params)
        self.L1 = sum(T.sum(abs(item)) for item in self.params)

class rnnlm(object):

    def __init__(self, vocab_size, n_emb_lstm, n_emb_struct, n_emb_share, n_hids, n_shids, n_structs, dropout, **kwargs):
        self.n_hids = n_hids
        self.n_shids = n_shids # kwargs.pop('n_shids', n_hids)
        self.n_structs = n_structs
        self.n_emb_lstm = n_emb_lstm
        self.n_emb_struct = n_emb_struct
        self.n_emb_share = n_emb_share #kwargs.pop('n_emb_share', min(n_emb_lstm, n_emb_struct)/2)
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.params = []
        self.layers = []

    def apply(self, sentence, sentence_mask, use_noise=1):
        n_emb_lstm = self.n_emb_lstm
        n_emb_struct = self.n_emb_struct
        n_emb_share = self.n_emb_share

        src = sentence[:-1]
        src_mask = sentence_mask[:-1]
        tgt = sentence[1:]
        tgt_mask = sentence_mask[1:]

        emb_lstm_range = T.arange(n_emb_lstm)
        table = lookup_table(n_emb_lstm, self.vocab_size, name='Wemb')
        state_below = table.apply(src, emb_lstm_range)
        self.layers.append(table)

        if self.dropout < 1.0:
            state_below = dropout_layer(state_below, use_noise, self.dropout)

        rnn = LSTM(n_emb_lstm, self.n_hids)
        hiddens, cells = rnn.apply(state_below, src_mask)
        #hiddens = rnn.merge_out(state_below, src_mask)
        self.layers.append(rnn)

        if True:
            if self.dropout < 1.0:
                hiddens = dropout_layer(hiddens, use_noise, self.dropout)

            rnn1 = LSTM(self.n_hids, self.n_hids)
            hiddens, cells = rnn1.apply(hiddens, src_mask)
            #hiddens = rnn.merge_out(state_below, src_mask)
            self.layers.append(rnn1)

        if True:
            if self.dropout < 1.0:
                hiddens = dropout_layer(hiddens, use_noise, self.dropout)

            rnnp = rnn_pyramid_layer(self.n_hids, n_emb_lstm, self.n_hids)
            hiddens,cells,structs,pyramid = rnnp.apply(hiddens, state_below, src_mask)
            self.layers.append(rnnp)
            #self.structs = structs
            self.rnn_len = rnnp.n_steps
        self.sent_len = sentence.shape[0]

        if True:
            maxout = maxout_layer()
            states = T.concatenate([state_below, hiddens], axis=2)
            maxout_n_fold = 2
            hiddens = maxout.apply(states, n_emb_lstm + self.n_hids, self.n_hids, src_mask, maxout_n_fold)
            self.layers.append(maxout)

        if self.dropout < 1.0:
            hiddens = dropout_layer(hiddens, use_noise, self.dropout)

        logistic_layer = LogisticRegression(hiddens, self.n_hids, self.vocab_size)
        self.layers.append(logistic_layer)

        self.cost = logistic_layer.cost(tgt, tgt_mask)

        for layer in self.layers:
            self.params.extend(layer.params)

        self.L2 = sum(T.sum(item ** 2) for item in self.params)
        self.L1 = sum(T.sum(abs(item)) for item in self.params)

def test(test_fn , tst_stream):
    sums = 0
    case = 0
    case_wrong = 0
    for sentence, sentence_mask in tst_stream.get_epoch_iterator():
        cost = test_fn(sentence.T, sentence_mask.T, 0)
        sums += cost[0]
        #case += sentence_mask[1:].sum()
        case += sentence_mask[:,1:].sum()
        case_wrong += sentence_mask[1:].sum()
    ppl = numpy.exp(sums/case)
    ppl_wrong = numpy.exp(sums/case_wrong)
    return ppl, ppl_wrong

def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
            numpy.isnan(output[0]).any()):
            print('*** NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break

def detect_nan_test(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
            numpy.isnan(output[0]).any()):
            print('*** test NaN detected ***')
            theano.printing.debugprint(node)
            print('Inputs : %s' % [input[0] for input in fn.inputs])
            print('Outputs: %s' % [output[0] for output in fn.outputs])
            break

def soft_clipping_curve(cur_epoch, last_curve_epoch, clip_begin, clip_end):
    cur_clip = clip_end
    if cur_epoch < last_curve_epoch:
        clip_diff = clip_begin - clip_end
        clip_step = clip_diff / last_curve_epoch
        cur_clip = clip_begin - cur_epoch * clip_step
    return cur_clip

if __name__=='__main__':
    import configurations
    cfig = getattr(configurations, 'get_config_penn')()
    if True:
    #for batch_size in [100]:
    #for nhids in [64 * i for i in [1,2,4,8]]:
        #for nemb in [64 * i for i in [1,2,4,8]]:
            logger.info("\nModel options:\n{}".format(pprint.pformat(cfig)))

            sentence = T.lmatrix()
            sentence_mask = T.matrix()
            use_noise = T.iscalar()
            lm = rnnlm(**cfig)
            lm.apply(sentence, sentence_mask, use_noise)

            #struct_num = lm.structs.shape[0]
            #struct_num = lm.rnn_len
            struct_num = lm.sent_len

            cost_sum = lm.cost
            cost_mean = lm.cost/sentence.shape[1]

            params = lm.params
            regular = lm.L1 * 1e-5 + lm.L2 * 1e-5

            grads = T.grad(cost_mean, params)

            hard_clipping = cfig['hard_clipping']
            soft_clipping = T.fscalar()
	    #if cfig['soft_clipping_begin'] > 0.:
            skip_nan_batch = 0
            grads, nan_num, inf_num = step_clipping(params, grads, soft_clipping, cfig['shrink_scale_after_skip_nan_grad'], cfig['skip_nan_grad'])

            updates = adadelta(params, grads, hard_clipping)
            vs = DStream(datatype='valid', config=cfig)
            ts = DStream(datatype='test', config=cfig)

	    #mode = theano.compile.MonitorMode(post_func=detect_nan).excluding('local_elemwise_fusion', 'inplace')
	    #mode_test = theano.compile.MonitorMode(post_func=detect_nan_test).excluding('local_elemwise_fusion', 'inplace')
            #fn = theano.function([sentence, sentence_mask, use_noise], [cost_mean, struct_num], updates=updates, mode=mode)
            #test_fn = theano.function([sentence, sentence_mask, use_noise], [cost_sum], mode=mode_test)

	    #mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True)
            #fn = theano.function([sentence, sentence_mask, use_noise], [cost_mean, struct_num], updates=updates, mode=mode)
            #test_fn = theano.function([sentence, sentence_mask, use_noise], [cost_sum], mode=mode)

            fn = theano.function([sentence, sentence_mask, use_noise, soft_clipping], [cost_mean, struct_num, nan_num, inf_num], updates=updates)
            test_fn = theano.function([sentence, sentence_mask, use_noise], [cost_sum])

            start_time = datetime.now()
            cur_time = start_time
            print ('training start at {}'.format(start_time))

            valid_errs = []
            test_errs = []
            patience = 200
            bad_counter = 0

            checked_sum = 0
            checked_sum_all = 0
            check_freq = 5000
            for epoch in range(500):
                #logger.info('{} epoch has been tackled with {} bad;'.format(epoch, bad_counter))
                ds = DStream(datatype='train', config=cfig)
                for data, mask in ds.get_epoch_iterator():
                    #print "data: ", data.T[0,:];
                    #print "mask: ", mask.T[0,:];
                    if cfig['drop_last_batch_if_small'] and (0.0 + len(data)) / cfig['batch_size'] < 0.95:
                        #logger.info('drop batch with: {}/{} ratio'.format(len(data), cfig['batch_size']))
                        pass # FIXME any idea to identify the last batch?
                    else:
                        cur_clip = soft_clipping_curve(epoch, cfig['soft_clipping_epoch'], cfig['soft_clipping_begin'], cfig['soft_clipping_end'])
                        cur_batch_time = datetime.now()                            
                        c, sn, grad_nan_num, grad_inf_num = fn(data.T, mask.T, 1, cur_clip)
                        batch_elasped_seconds = (datetime.now() - cur_batch_time).total_seconds() 
                        logger.info('grad nan/inf num: {} {} at epoch {} cost {}'.format(grad_nan_num, grad_inf_num, epoch, batch_elasped_seconds))
                        #logger.info('lm.cost.mean: {}, {}'.format(c, sn))
                        #valid_err=test(test_fn, vs)
                        #test_err=test(test_fn, ts)
		        #logger.info('valid, test ppl : {}, {}'.format(valid_err, test_err))
                        checked_sum_all += cfig['batch_size']
                        if checked_sum_all < cfig['check_low_freq_count']:
                            check_freq = cfig['check_low_freq']
                        else:
                            check_freq = cfig['check_freq']
                        if checked_sum < check_freq:
                            checked_sum += cfig['batch_size']
                            #print checked_sum/cfig['check_freq'], '\r'
                            #print "."
                            # FIXME : if train size < check_freq, there will be no check point
                        else:
                            valid_err, valid_err_wrong =test(test_fn, vs)
                            test_err, test_err_wrong =test(test_fn, ts)
                            checked_sum = 0
                            valid_errs.append(valid_err)
                            test_errs.append(test_err)
                            if valid_err <= numpy.array(valid_errs).min():
                                bad_counter = 0
                            if len(valid_errs) > patience and valid_err >= \
                                numpy.array(valid_errs)[:-patience].min():       
                                bad_counter += 1                                 
                            valid_min = numpy.min(valid_errs)
                            valid_min_idx = numpy.argmin(valid_errs)
                            valid_min_test = test_errs[valid_min_idx]
                            pre_time = cur_time
                            cur_time = datetime.now()                            
                            elasped_minutes = (cur_time - start_time).total_seconds() / 60.
                            batch_elasped_seconds = (cur_time - pre_time).total_seconds()
                            #logger.info('{:>3} epoch {:>2} bad test/valid ppl {:>5.2f} {:>5.2f} i:m:tvbest {:>3} {:>3} {:>5.2f} {:>5.2f} {:>6.2f} min'.\
                            print ('{:>3} epoch {:>2} bad test/valid(wrong) ppl {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} i:m:tvbest {:>3} {:>3} {:>5.2f} {:>5.2f} batch {:>4.0f}s, all {:>5.1f}m nan {} inf {}'.\
                                format(epoch, bad_counter, test_err, valid_err, test_err_wrong, valid_err_wrong, len(valid_errs)-1, valid_min_idx, valid_min_test, valid_min, batch_elasped_seconds, elasped_minutes, grad_nan_num, grad_inf_num));
                            sys.stdout.flush()
                            if bad_counter > patience:                           
                                print "Early Stop! inner loop"                   
                                break                                            
                if bad_counter > patience:                                       
                    print "Early Stop! outter loop"                              
                    break                                                        

            valid_min = numpy.min(valid_errs)
            valid_min_idx = numpy.argmin(valid_errs)
            valid_min_test = test_errs[valid_min_idx]
            test_min = numpy.min(test_errs)
            test_min_idx = numpy.argmin(test_errs)
            cur_cfig = str(valid_min_idx)  + ':' + str(valid_min)
            cur_cfig += ',' + str(valid_min_test) + ','
            cur_cfig += str(test_min_idx)  + ':' + str(test_min)
            print '#== epoch:valid_min,valid_min_test,epoch:test_min: ' + cur_cfig


