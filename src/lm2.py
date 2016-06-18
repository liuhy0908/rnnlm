#encoding=utf8
import theano
import theano.tensor as T
from model import NormalRNN , MorphRNN , MorphStruct , GRU, LSTM, FLSTM, lookup_table,dropout_layer, maxout_layer, rnn_pyramid_layer, LogisticRegression
from utils import adadelta, step_clipping
from stream_morph import DStream
import numpy
from datetime import datetime,timedelta
import time
import sys
import logging
import pprint

logger = logging.getLogger(__name__)

class rnnlm_morph(object):
    def __init__(self, vocab_size, morph_size , n_emb_lstm , n_emb_morph , n_hids, dropout, **kwargs):
        self.n_hids = n_hids
        self.n_emb_lstm = n_emb_lstm
        self.n_emb_morph = n_emb_morph
        self.vocab_size = vocab_size
        self.morph_size = morph_size
        self.dropout = dropout
        self.params = []
        self.layers = []

    def apply(self, sentence, sentence_mask, sentence_morph, sentence_morph_mask, use_noise=1):
        """
            sentence : sentence * batch
            sentence_morph : sentence * batch * morph
        """
        n_emb_lstm = self.n_emb_lstm
        n_emb_morph = self.n_emb_morph

        src = sentence[:-1]
        src_mask = sentence_mask[:-1]
        src_morph = sentence_morph[:-1]
        src_morph_mask = sentence_morph_mask[:-1]
        tgt = sentence[1:]
        tgt_mask = sentence_mask[1:]

        emb_lstm_range = T.arange(n_emb_lstm)
        #word lookup table
        table = lookup_table(n_emb_lstm, self.vocab_size, name='Wemb')
        state_below = table.apply(src, emb_lstm_range)
        self.layers.append(table)

        emb_morph_range = T.arange(n_emb_morph)
        #morph lookup table
        table_morph = lookup_table(n_emb_morph, self.morph_size, name='Memb')
        state_below_morph = table_morph.apply(src_morph, emb_morph_range)
        self.layers.append(table_morph)

        if self.dropout < 1.0:
            state_below = dropout_layer(state_below, use_noise, self.dropout)
            state_below_morph = dropout_layer(state_below_morph, use_noise, self.dropout)

        #rnn = LSTM(n_emb_lstm, self.n_hids)
        #hiddens , cells  = rnn.apply(state_below, src_mask)
        #self.layers.append(rnn)
        #if self.dropout < 1.0:
        #    hiddens = dropout_layer(hiddens, use_noise, self.dropout)
        #rnn2 = LSTM(self.n_hids, self.n_hids)
        #hiddens , cells = rnn2.apply(hiddens , src_mask)
        #self.layers.append(rnn2)

        morphStruct = MorphStruct()
        morph_out = morphStruct.apply(state_below_morph , src_morph_mask)
        #self.layers.append(morphStruct)

        rnn = MorphRNN(n_emb_lstm , n_emb_morph , self.n_hids)
        hiddens  = rnn.apply(state_below, src_mask , morph_out)
        self.layers.append(rnn)
        #rnn = NormalRNN(n_emb_lstm , self.n_hids)
        #hiddens  = rnn.apply(state_below, src_mask)
        #self.layers.append(rnn)

        if False:
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

def test(test_fn , tst_stream , tst_morph , tst_morph_mask):
    sums = 0
    case = 0
    for data_tuple , data_morph , mask_morph in zip(tst_stream.get_epoch_iterator() , tst_morph , tst_morph_mask):
        data , mask = data_tuple
        data_morph = data_morph.transpose((1 , 0 , 2))
        mask_morph = mask_morph.transpose((1 , 0 , 2))
        cost = test_fn(data.T, mask.T, data_morph, mask_morph, 0)
        sums += cost[0]
        case += mask[:,1:].sum()
    ppl = numpy.exp(sums/case)
    return ppl

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
    cfig = getattr(configurations, 'get_config_morph')()
    logger.info("\nModel options:\n{}".format(pprint.pformat(cfig)))

    sentence = T.lmatrix()
    sentence_mask = T.matrix()
    sentence_morph = T.ltensor3()
    sentence_morph_mask = T.tensor3()

    use_noise = T.iscalar()
    lm = rnnlm_morph(**cfig)
    lm.apply(sentence, sentence_mask, sentence_morph, sentence_morph_mask, use_noise)

    cost_sum = lm.cost
    cost_mean = lm.cost/sentence.shape[1]
    params = lm.params
    regular = lm.L1 * 1e-5 + lm.L2 * 1e-5
    grads = T.grad(cost_mean, params)

    hard_clipping = cfig['hard_clipping']
    soft_clipping = T.fscalar()
    skip_nan_batch = 0
    grads, nan_num, inf_num = step_clipping(params, grads, soft_clipping, cfig['shrink_scale_after_skip_nan_grad'], cfig['skip_nan_grad'])

    updates = adadelta(params, grads, hard_clipping)
    vs , vs_morph , vs_morph_mask = DStream(datatype='valid', config=cfig)
    ts , ts_morph , ts_morph_mask = DStream(datatype='test', config=cfig)

    fn = theano.function([sentence, sentence_mask, sentence_morph, sentence_morph_mask, use_noise, soft_clipping], [cost_mean, nan_num, inf_num], updates=updates , on_unused_input='ignore')
    test_fn = theano.function([sentence, sentence_mask, sentence_morph, sentence_morph_mask, use_noise], [cost_sum] , on_unused_input='ignore')

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
        ds , ds_morph , ds_morph_mask = DStream(datatype='train', config=cfig)
        for data_tuple , data_morph , mask_morph in zip(ds.get_epoch_iterator() , ds_morph , ds_morph_mask):
            data , mask = data_tuple
            #print data , data.shape
            if cfig['drop_last_batch_if_small'] and (0.0 + len(data)) / cfig['batch_size'] < 0.95:
                #logger.info('drop batch with: {}/{} ratio'.format(len(data), cfig['batch_size']))
                pass # FIXME any idea to identify the last batch?
            else:
                cur_clip = soft_clipping_curve(epoch, cfig['soft_clipping_epoch'], cfig['soft_clipping_begin'], cfig['soft_clipping_end'])
                cur_batch_time = datetime.now()
                data_morph = data_morph.transpose((1 , 0 , 2))
                mask_morph = mask_morph.transpose((1 , 0 , 2))
                #print data.T.shape , data_morph.shape
                c, grad_nan_num, grad_inf_num = fn(data.T, mask.T, data_morph , mask_morph , 1, cur_clip)
                batch_elasped_seconds = (datetime.now() - cur_batch_time).total_seconds()
                logger.info('grad nan/inf num: {} {} at epoch {} cost {}'.format(grad_nan_num, grad_inf_num, epoch, batch_elasped_seconds))
                #checked_sum_all记录总共训练的句子数
                checked_sum_all += cfig['batch_size']
                if checked_sum_all < cfig['check_low_freq_count']:
                    check_freq = cfig['check_low_freq']
                else:
                    check_freq = cfig['check_freq']
                #表示每训练check_freq个句子后对valid和test算ppl
                #check_freq记录这一轮已经训练的句子树
                if checked_sum < check_freq:
                    checked_sum += cfig['batch_size']
                    # FIXME : if train size < check_freq, there will be no check point
                else:
                    valid_err = test(test_fn, vs , vs_morph , vs_morph_mask)
                    test_err = test(test_fn, ts , ts_morph , ts_morph_mask)
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
                    print ('{:>3} epoch {:>2} bad test/valid ppl {:>5.2f} {:>5.2f} i:m:tvbest {:>3} {:>3} {:>5.2f} {:>5.2f} batch {:>4.0f}s, all {:>5.1f}m nan {} inf {}'.\
                        format(epoch, bad_counter, test_err, valid_err, len(valid_errs)-1, valid_min_idx, valid_min_test, valid_min, batch_elasped_seconds, elasped_minutes, grad_nan_num, grad_inf_num));
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


