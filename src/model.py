import theano
import theano.tensor as T
import numpy
from utils import param_init, repeat_x, concatenate
from theano.tensor.nnet import categorical_crossentropy
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from unit import GRU as GRU_Z
from unit import ATTENTION as ATT

class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = param_init().uniform((n_in, n_out), 'W_logreg') # FIXME : uniform weight?
        # initialize the baises b as a vector of n_out 0s
        self.b = param_init().constant((n_out, ), 'b_logreg')

        # compute vector of class-membership probabilities in symbolic form
        energy = theano.dot(input, self.W) + self.b
        if energy.ndim == 3:
            energy_exp = T.exp(energy - T.max(energy, 2, keepdims=True))
            pmf = energy_exp / energy_exp.sum(2, keepdims=True)
            #pmf = T.nnet.softmax(energy.reshape([energy.shape[0]*energy.shape[1], energy.shape[2]])).reshape([energy.shape[0],energy.shape[1],energy.shape[2]])
	    #pmf = T.exp(energy)/(T.exp(energy).sum(2,keepdims=True))
        else:
            pmf = T.nnet.softmax(energy)
	    #pmf = T.exp(energy)/(T.exp(energy).sum(1,keepdims=True))

        self.p_y_given_x = pmf

        # compute prediction as class whose probability is maximal in
        # symbolic form

        # parameters of the model
        self.params = [self.W, self.b]

    def cost(self, targets, mask=None):
        prediction = self.p_y_given_x

        if prediction.ndim == 3:
        # prediction = prediction.dimshuffle(1,2,0).flatten(2).dimshuffle(1,0)
            prediction_flat = prediction.reshape(((prediction.shape[0] *
                                                prediction.shape[1]),
                                                prediction.shape[2]), ndim=2)
            targets_flat = targets.flatten()
            mask_flat = mask.flatten()
            ce = categorical_crossentropy(prediction_flat, targets_flat) * mask_flat
        else:
            assert mask is None
            ce = categorical_crossentropy(prediction, targets)
        return T.sum(ce)

    def errors(self, y):
        y_pred = T.argmax(self.p_y_given_x, axis=-1)
        if y.ndim == 2:
            y = y.flatten()
            y_pred = y_pred.flatten()
        return T.sum(T.neq(y, y_pred))


class NormalRNN(object):
    """
        normal rnn
        h_t = h_tm1 * W_hh + x_t * W_xh + b
    """
    def __init__(self , n_in , n_hids , with_contex=False , **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids' , n_hids)
        self._init_params()

    def _init_params(self):
        size_xh = (self.n_in , self.n_hids)
        size_hh = (self.n_hids , self.n_hids)
        self.W_xh = param_init().uniform(size_xh)
        self.W_hh = param_init().uniform(size_hh)
        self.b = param_init().constant((self.n_hids,))
        self.params = [self.W_xh , self.W_hh ,
                       self.b]

    def _step(self , x_t , x_m , h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t , self.W_xh)
                             + T.dot(h_tm1 , self.W_hh) + self.b)
        h_t = x_m[:,None] * h_t + (1. - x_m[:,None]) * h_tm1
        return h_t

    def apply(self, state_below, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError
        if init_state is None:
            init_state = T.alloc(numpy.float32(0.) , batch_size , self.n_hids)
        rval , updates = theano.scan(self._step,
                                     sequences=[state_below , mask_below],
                                     outputs_info=[init_state],
                                     n_steps=n_steps)
        self.output = rval
        return self.output

class MorphRNN(object):
    """
        morph rnn
        h_t = h_tm1 * W_hh + x_t * W_xh + m_tm1 * W_mh + b
    """
    def __init__(self , n_in , n_morph_in, n_hids , **kwargs):
        self.n_in = n_in
        self.n_morph_in = n_morph_in
        self.n_hids = n_hids
        self._init_params()

    def _init_params(self):
        size_xh = (self.n_in , self.n_hids)
        size_mh = (self.n_morph_in , self.n_hids)
        size_hh = (self.n_hids , self.n_hids)
        self.W_xh = param_init().uniform(size_xh)
        self.W_hh = param_init().uniform(size_hh)
        self.W_mh = param_init().uniform(size_mh)
        self.b = param_init().constant((self.n_hids,))
        self.params = [self.W_xh , self.W_hh ,
                       self.W_mh , self.b]

    def _step(self , x_t , x_m , m_t , h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t , self.W_xh) + T.dot(h_tm1 , self.W_hh)
                             + T.dot(m_t , self.W_mh) + self.b)
        h_t = x_m[:,None] * h_t + (1. - x_m[:,None]) * h_tm1
        return h_t

    def apply(self, state_below, mask_below, morph_out ,init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError
        if init_state is None:
            init_state = T.alloc(numpy.float32(0.) , batch_size , self.n_hids)
        rval , updates = theano.scan(self._step,
                                     sequences=[state_below , mask_below , morph_out],
                                     outputs_info=[init_state],
                                     n_steps=n_steps)
        self.output = rval
        return self.output

class MorphStruct(object):
    """
        morph struct
    """
    def __init__(self):
        pass

    def apply(self, state_below_morph , mask_morph ,init_state=None, context=None):
        """
            state_below_morph : sentence * batch * morph * n_emb_morph
        """
        #h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
        output = state_below_morph.sum(2)
        return output


class MorphStructRNN(object):
    """
        morph struct with rnn
    """
    def __init__(self , n_emb_morph , n_hids):
        self.n_hids = n_hids
        self.lstm = LSTM(n_emb_morph, self.n_hids)
        self.params = []
        self.params.extend(self.lstm.params)
    def apply(self , state_below , mask_morph):
        """
            state_below : sentence * batch * morph * n_emb_morph
            return : sentence * batch * n_emb
        """
        newshape = [state_below.shape[2] , state_below.shape[0] * state_below.shape[1] , state_below.shape[3]]
        outshape = [state_below.shape[0] , state_below.shape[1] , self.n_hids]
        state_below = state_below.transpose((2 , 0 , 1 , 3)).reshape(newshape)
        maskshape = [mask_morph.shape[2] , mask_morph.shape[0] * mask_morph.shape[1]]
        mask_morph = mask_morph.transpose(2 , 0 , 1).reshape(maskshape)
        '''
            lstm input : sentence * batch * emb
            lstm output : sentence * batch * n_hids
            real input : morph * (sentence * batch) * emb_morph
            mask_morph : morph * (sentence * batch)
        '''
        hiddens , cells  = self.lstm.apply(state_below, mask_morph)
        '''
        if state_below.ndim == 1:
            pass
        else:
            print hiddens.flatten().shape
            print hiddens[:,:,-1].transpose(1 , 0).reshape(outshape).flatten().shape
            raise NotImplementedError
        '''
        hiddens = hiddens[-1,:,:].reshape(outshape)
        # hiddens : sentence * batch * n_hids
        return hiddens


class GRU(object):

    def __init__(self, n_in, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xz = param_init().uniform(size_xh)
        self.W_xr = param_init().uniform(size_xh)
        self.W_xh = param_init().uniform(size_xh)

        self.W_hz = param_init().orth(size_hh)
        self.W_hr = param_init().orth(size_hh)
        self.W_hh = param_init().orth(size_hh)

        self.b_z = param_init().constant((n_hids,))
        self.b_r = param_init().constant((n_hids,))
        self.b_h = param_init().constant((n_hids,))

        self.params = [self.W_xz, self.W_xr, self.W_xh,
                       self.W_hz, self.W_hr, self.W_hh,
                       self.b_z, self.b_r, self.b_h]

        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().uniform(size_ch)
            self.W_cr = param_init().uniform(size_ch)
            self.W_ch = param_init().uniform(size_ch)
            self.W_c_init = param_init().uniform(size_ch)

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]

    def _step_forward_with_context(self, x_t, x_m, h_tm1, c_z, c_r, c_h):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + c_z + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + c_r + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) + c_h +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t


    def _step_forward(self, x_t, x_m, h_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_x: contex of the rnn
        '''
        z_t = T.nnet.sigmoid(T.dot(x_t, self.W_xz) +
                             T.dot(h_tm1, self.W_hz) + self.b_z)

        r_t = T.nnet.sigmoid(T.dot(x_t, self.W_xr) +
                             T.dot(h_tm1, self.W_hr) + self.b_r)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) +
                         self.b_h)
        h_t = (1 - z_t) * h_tm1 + z_t * can_h_t

        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t

    def apply(self, state_below, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError
        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(self._step_forward_with_context,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            rval, updates = theano.scan(self._step_forward,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        n_steps=n_steps
                                        )
        self.output = rval
        return self.output

    def merge_out(self, state_below, mask_below, context=None):
        hiddens = self.apply(state_below, mask_below, context=context)
        if context is None:
            msize = self.n_in + self.n_hids
            osize = self.n_hids
            combine = T.concatenate([state_below, hiddens], axis=2)
        else:
            msize = self.n_in + self.n_hids + self.c_hids
            osize = self.n_hids
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)

        self.W_m = param_init().uniform((msize, osize*2))
        self.b_m = param_init().constant((osize*2,))
        self.params += [self.W_m, self.b_m]

        merge_out = theano.dot(combine, self.W_m) + self.b_m
        merge_max = merge_out.reshape((merge_out.shape[0],
                                       merge_out.shape[1],
                                       merge_out.shape[2]/2,
                                       2), ndim=4).max(axis=3)
        return merge_max * mask_below[:, :, None]

class LSTM(object):

    def __init__(self, n_in, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xi = param_init().normal(size_xh)
        self.W_xf = param_init().normal(size_xh)
        self.W_xo = param_init().normal(size_xh)
        self.W_xc = param_init().normal(size_xh)

        self.W_hi = param_init().orth(size_hh)
        self.W_hf = param_init().orth(size_hh)
        self.W_ho = param_init().orth(size_hh)
        self.W_hc = param_init().orth(size_hh)

        self.b_i = param_init().constant((n_hids,))
        self.b_f = param_init().constant((n_hids,), scale=1.0) #  Jozefowicz et al. 2015
        self.b_o = param_init().constant((n_hids,))
        self.b_c = param_init().constant((n_hids,))

        self.params = [self.W_xi, self.W_xf, self.W_xo, self.W_xc,
                       self.W_hi, self.W_hf, self.W_ho, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c]

        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().uniform(size_ch)
            self.W_cr = param_init().uniform(size_ch)
            self.W_ch = param_init().uniform(size_ch)
            self.W_c_init = param_init().uniform(size_ch)

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]

    def _step_forward(self, x_t, x_m, h_tm1, c_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_tm1: previous memory cell
        '''
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) +
                             T.dot(h_tm1, self.W_hi) + self.b_i)

        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) +
                             T.dot(h_tm1, self.W_hf) + self.b_f)

        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) +
                             T.dot(h_tm1, self.W_ho) + self.b_o)

        c_t = T.tanh(T.dot(x_t, self.W_xc) +
                             T.dot(h_tm1, self.W_hc) + self.b_c)

        c_t = f_t * c_tm1 + i_t * c_t
        c_t = x_m[:, None] * c_t + (1. - x_m[:, None]) * c_tm1

        h_t = o_t * T.tanh(c_t)
        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t, c_t

    def apply(self, state_below, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError


        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(self._step_forward_with_context,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                init_state1 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            rval, updates = theano.scan(self._step_forward,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state, init_state1],
                                        n_steps=n_steps
                                        )
        self.output = rval
        return self.output

    def merge_out(self, state_below, mask_below, context=None):
        hiddens, memo_cells = self.apply(state_below, mask_below, context=context)
        if context is None:
            msize = self.n_in + self.n_hids
            osize = self.n_hids
            combine = T.concatenate([state_below, hiddens], axis=2)
        else:
            msize = self.n_in + self.n_hids + self.c_hids
            osize = self.n_hids
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)

        self.W_m = param_init().normal((msize, osize*2))
        self.b_m = param_init().constant((osize*2,))
        self.params += [self.W_m, self.b_m]

        merge_out = theano.dot(combine, self.W_m) + self.b_m
        merge_max = merge_out.reshape((merge_out.shape[0],
                                       merge_out.shape[1],
                                       merge_out.shape[2]/2,
                                       2), ndim=4).max(axis=3)
        return merge_max * mask_below[:, :, None]


class FLSTM(object):

    def __init__(self, n_in, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xi = param_init().normal(size_xh)
        self.W_xf = param_init().normal(size_xh)
        self.W_xo = param_init().normal(size_xh)
        self.W_xc = param_init().normal(size_xh)

        self.W_hi = param_init().orth(size_hh)
        self.W_hf = param_init().orth(size_hh)
        self.W_ho = param_init().orth(size_hh)
        self.W_hc = param_init().orth(size_hh)

        self.b_i = param_init().constant((n_hids,))
        self.b_f = param_init().constant((n_hids,), scale=1.0) #  Jozefowicz et al. 2015
        self.b_o = param_init().constant((n_hids,))
        self.b_c = param_init().constant((n_hids,))

        self.params = [self.W_xi, self.W_xf, self.W_xo, self.W_xc,
                       self.W_hi, self.W_hf, self.W_ho, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c]

        #self.attention = ATT(self.n_hids)
        #self.params.extend(self.attention.params)
        self.W_at = param_init().normal(size_hh)
        self.W_at2 = param_init().normal(size_hh)
        self.W_U_att = param_init().normal((n_hids , 1))
        self.W_comb = param_init().normal(size_hh)
        self.W_comb2 = param_init().normal(size_hh)
        #self.W_comb3 = param_init().normal(size_hh)
        self.b_at = param_init().constant((n_hids,))
        self.b_c_att = param_init().constant((1,))
        self.b_comb = param_init().constant((n_hids,))
        self.params = self.params + [self.W_at , self.b_at,
                                     self.W_at2 ,
                                     self.W_U_att , self.b_c_att,
                                     self.W_comb , self.W_comb2 , self.b_comb]
        #self.params = self.params + [self.W_comb , self.W_comb2 , self.b_comb]
        #self.params = self.params + [self.W_comb , self.W_comb2 , self.W_comb3 , self.b_comb]
    def _step_forward(self, x_t, x_m, t , h_tm1, c_tm1 , his):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state. batch * hids
        c_tm1: previous memory cell
        his : seq * batch * hids
        '''
        #tmp = T.switch(T.gt(t , 0) , his[:t,:,:] , init2)
        #tmp = T.switch(T.gt(t , 0) , t , 1)
        #pre_ctx = T.dot(his[:tmp,:,:] , self.W_at) + self.b_at
        pre_ctx = T.dot(his[:t,:,:] , self.W_at) + self.b_at
        #pre_ctx : seq * batch * hids
        pstate = T.dot(h_tm1 , self.W_at2)
        #pstate : batch * hids
        pstate = T.tanh(pstate[None , : , :] + pre_ctx)
        #pstate : seq * batch * hids
        alpha = T.dot(pstate , self.W_U_att ) + self.b_c_att
        #alpha : seq * batch * 1
        alpha = alpha.reshape([alpha.shape[0] , alpha.shape[1]])
        #alpha : seq * batch
        alpha = T.exp(alpha)
        alpha = alpha / alpha.sum(0 , keepdims=True)
        #his : seq * batch * hids
        #alpha : seq * batch
        #ctx_psum = (his[:tmp,:,:] * alpha[:,:,None]).sum(0)
        ctx_psum = (his[:t,:,:] * alpha[:,:,None]).sum(0)
        #ctx_psum : batch * hids

        #pre_ctx = T.dot(his[t-1,:,:] , self.W_at) + self.b_at
        #ctx_psum , alphaT = self.attention.apply(h_tm1 , pre_ctx , his[0:t,:,:])
        #tmp = T.switch(T.ge(t , 2) , t , 2)
        #pre_ctx = T.dot(his[tmp-2:t] , self.W_at) + self.b_at
        #ctx_psum , alphaT = self.attention.apply(h_tm1 , pre_ctx , his[tmp-2:t])

        h_tm1 = T.nnet.sigmoid(T.dot(ctx_psum , self.W_comb) +
                             T.dot(h_tm1 , self.W_comb2) + self.b_comb)
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) +
                             T.dot(h_tm1, self.W_hi) + self.b_i)
        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) +
                             T.dot(h_tm1, self.W_hf) + self.b_f)
        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) +
                             T.dot(h_tm1, self.W_ho) + self.b_o)
        c_t = T.tanh(T.dot(x_t, self.W_xc) +
                             T.dot(h_tm1, self.W_hc) + self.b_c)
        c_t = f_t * c_tm1 + i_t * c_t
        c_t = x_m[:, None] * c_t + (1. - x_m[:, None]) * c_tm1
        h_t = o_t * T.tanh(c_t)
        h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1
        return h_t, c_t

    def apply(self, state_below, his , mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError
        if init_state is None:
            init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            init_state1 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
        pre_t = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
        his = T.concatenate([pre_t[None,:,:] , his] , axis = 0)
        step_idx = T.arange(1 , n_steps + 1)
        rval, updates = theano.scan(self._step_forward,
                                    sequences=[state_below, mask_below ,step_idx],
                                    outputs_info=[init_state, init_state1],
                                    non_sequences=[his],
                                    n_steps=n_steps
                                    )
        self.output = rval
        return self.output

class FLSTM_bp(object):
    def __init__(self, n_in, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xi = param_init().normal(size_xh)
        self.W_xf = param_init().normal(size_xh)
        self.W_xo = param_init().normal(size_xh)
        self.W_xc = param_init().normal(size_xh)

        self.W_hi = param_init().orth(size_hh)
        self.W_hf = param_init().orth(size_hh)
        self.W_ho = param_init().orth(size_hh)
        self.W_hc = param_init().orth(size_hh)

        self.b_i = param_init().constant((n_hids,))
        self.b_f = param_init().constant((n_hids,), scale=1.0) #  Jozefowicz et al. 2015
        self.b_o = param_init().constant((n_hids,))
        self.b_c = param_init().constant((n_hids,))

        self.params = [self.W_xi, self.W_xf, self.W_xo, self.W_xc,
                       self.W_hi, self.W_hf, self.W_ho, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c]

        self.attention = ATT(self.n_hids)
        self.W_at = param_init().normal(size_hh)
        self.W_comb = param_init().normal(size_hh)
        self.W_comb2 = param_init().normal(size_hh)
        self.W_comb3 = param_init().normal(size_hh)
        self.b_at = param_init().constant((n_hids,))
        self.b_comb = param_init().constant((n_hids,))

        #self.params.extend(self.attention.params)
        #self.params = self.params + [self.W_at , self.b_at,
        #                             self.W_comb , self.W_comb2 , self.b_comb]
        #self.params = self.params + [self.W_comb , self.W_comb2 , self.b_comb]
        self.params = self.params + [self.W_comb , self.W_comb2 , self.W_comb3 , self.b_comb]

    #def _step_forward(self, x_t, x_m, t , h_tm1, c_tm1 , h_tm2 , h_tm3):
    def _step_forward(self, x_t, x_m, t , h_tm1, c_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_tm1: previous memory cell
        '''
        h_tm1 = T.nnet.sigmoid(T.dot(h_tm1 , self.W_comb) +
                               T.dot(h_tm2 , self.W_comb2) +
                               T.dot(h_tm3 , self.W_comb3) + self.b_comb)
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) +
                             T.dot(h_tm1, self.W_hi) + self.b_i)

        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) +
                             T.dot(h_tm1, self.W_hf) + self.b_f)

        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) +
                             T.dot(h_tm1, self.W_ho) + self.b_o)

        c_t = T.tanh(T.dot(x_t, self.W_xc) +
                             T.dot(h_tm1, self.W_hc) + self.b_c)

        c_t = f_t * c_tm1 + i_t * c_t
        c_t = x_m[:, None] * c_t + (1. - x_m[:, None]) * c_tm1

        h_t = o_t * T.tanh(c_t)
        h_t = x_m[:, None] * h_t + (1. - x_m[:, None]) * h_tm1

        #pre_ctx = T.dot(his[0:t-1] , self.W_at) + self.b_at
        #ctx_psum , alphaT = self.attention.apply(h_t , pre_ctx , his[0:t-1])
        #h_2 = T.nnet.sigmoid(T.dot(ctx_psum , self.W_comb) +
        #                     T.dot(h_t , self.W_comb2) + self.b_comb)
        #h_2 = x_m[:,None] * h_2 + (1. - x_m[: , None]) * h_tm1
        #T.set_subtensor(his[t,:] , h_2)
        return h_t, c_t , h_tm1 , h_tm2

    def apply(self, state_below, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError
        if init_state is None:
            init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            init_state1 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            init_state2 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            init_state3 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
        init_his = T.alloc(numpy.float32(0.) , n_steps , batch_size , self.n_hids)
        step_idx = T.arange(n_steps)
        rval, updates = theano.scan(self._step_forward,
                                    sequences=[state_below, mask_below ,step_idx],
                                    outputs_info=[init_state, init_state1 , init_state2 , init_state3],
                                    non_sequences=[],
                                    n_steps=n_steps
                                    )
        self.output = rval
        return self.output

class SLSTM(object):

    def __init__(self, n_emb_lstm, n_emb_struct, n_emb_share, n_hids, n_shids, n_structs, with_contex=False, **kwargs):
        self.n_hids = n_hids
        self.n_shids = n_shids # kwargs.pop('n_shids', n_hids)
        self.n_structs = n_structs
        self.n_emb_lstm = n_emb_lstm
        self.n_emb_struct = n_emb_struct
        self.n_emb_share = n_emb_share # kwargs.pop('n_emb_share', min(n_emb_lstm, n_emb_struct)/2)

        self.n_structs_sum = sum(range(1, n_structs+1))
        self.with_contex = with_contex
        self._init_params()

    def _init_params(self):
        n_emb_lstm = self.n_emb_lstm
        n_emb_struct = self.n_emb_struct
        n_emb_share = self.n_emb_share

        n_hids = self.n_hids
        n_shids = self.n_shids
        n_structs = self.n_structs
        n_structs_sum = self.n_structs_sum

        size_xh = (n_emb_lstm, n_hids)
        size_hh = (n_hids, n_hids)
        size_hs = (n_hids, n_shids)
        size_sh = (n_shids, n_hids)
        size_hns= (n_hids, n_shids * n_structs )
        size_sess= (n_emb_struct,  n_shids * n_structs_sum)

        self.W_xi = param_init().normal(size_xh, 'W_xi')
        self.W_xf = param_init().normal(size_xh, 'W_xf')
        self.W_xo = param_init().normal(size_xh, 'W_xo')
        self.W_xc = param_init().normal(size_xh, 'W_xc')

        self.W_hi = param_init().orth(size_hh, 'W_hi')
        self.W_hf = param_init().orth(size_hh, 'W_hf')
        self.W_ho = param_init().orth(size_hh, 'W_ho')
        self.W_hc = param_init().orth(size_hh, 'W_hc')

        self.b_i = param_init().constant((n_hids,), 'b_i')
        self.b_f = param_init().constant((n_hids,), 'b_f')
        self.b_o = param_init().constant((n_hids,), 'b_o')
        self.b_c = param_init().constant((n_hids,), 'b_c')

        # embedding to structure proposal weights, biases
        self.W_xs = param_init().normal(size_sess, 'W_xs')
        self.b_xs = param_init().constant((n_structs * n_shids,), 'b_xs')

        # embedding to structure selection weights, biases
        self.W_xss = param_init().normal(size_sess, 'W_xss')
        self.b_xss = param_init().constant((n_structs * n_shids,), 'b_xss')

        # recurrent transformation weights for structure proposal and selection
        self.W_cs  = param_init().multi_orth(size_hns, 'W_cs')
        self.W_css = param_init().multi_orth(size_hns, 'W_css')

        # embedding to hidden state proposal weights, biases
        self.W_s_h = param_init().orth(size_sh, 'W_s_h')
        self.W_h_h = param_init().orth(size_hh, 'W_h_h')
        self.b_s_h = param_init().constant((n_hids,), 'b_s_h')


        self.params = [self.W_xi, self.W_xf, self.W_xo, self.W_xc, \
                       self.W_hi, self.W_hf, self.W_ho, self.W_hc, \
                       self.b_i,  self.b_f,  self.b_o,  self.b_c,  \
                                     self.W_xs,  self.W_xss,  \
                                     self.b_xs,  self.b_xss,  \
                                     self.W_cs,  self.W_css,  \
                                     self.W_s_h, self.W_h_h,  \
                                     self.b_s_h]
        #self.params = self.params + [self.W_xs,  self.W_xss,
        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().uniform(size_ch)
            self.W_cr = param_init().uniform(size_ch)
            self.W_ch = param_init().uniform(size_ch)
            self.W_c_init = param_init().uniform(size_ch)

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]


    def _step_forward(self, x_t, x_m, s, ss, h_tm1, c_tm1, s_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_tm1: previous memory cell
        '''
        i_t = T.nnet.sigmoid(T.dot(x_t, self.W_xi) +
                             T.dot(h_tm1, self.W_hi) + self.b_i)

        f_t = T.nnet.sigmoid(T.dot(x_t, self.W_xf) +
                             T.dot(h_tm1, self.W_hf) + self.b_f)

        o_t = T.nnet.sigmoid(T.dot(x_t, self.W_xo) +
                             T.dot(h_tm1, self.W_ho) + self.b_o)

        c_t = T.tanh(T.dot(x_t, self.W_xc) +
                             T.dot(h_tm1, self.W_hc) + self.b_c)

        c_t = f_t * c_tm1 + i_t * c_t
        c_t = x_m[:, None] * c_t + (1. - x_m[:, None]) * c_tm1

        # compute structure
        batch_size = x_t.shape[0]

        s = T.tanh(s + T.dot(c_t, self.W_cs))
        s = s.reshape([batch_size, self.n_structs, self.n_shids])

        ss = T.nnet.sigmoid(ss + T.dot(c_t, self.W_css))
        ss = T.nnet.softmax(ss.reshape([batch_size * self.n_structs, self.n_shids]))
        ss = ss.reshape([batch_size, self.n_structs, self.n_shids])

        s_t = s * ss
        s_t = s_t.sum(axis=1) # TODO: use s_tm1 in construct s
        s_t = x_m[:, None] * s_t + (1. - x_m[:, None])*s_tm1

        h_t = o_t * T.tanh(c_t)
        # include structure in construct hidden state
        h_t = T.tanh(T.dot(s_t, self.W_s_h) + T.dot(h_t, self.W_h_h)+ self.b_s_h)
        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t, c_t, s_t

    # utitlity function for transfer unsliced history state into correct order
    def shift_combine(self, inps):
        dim = self.n_shids
        ns = self.n_structs
        rval = inps[:,:,:ns*dim]
        end = ns
        for i in range(1, ns):
            s_len = ns - i
            begin = end
            end = begin + s_len
            rval = T.set_subtensor( rval[i:,:,i*dim:],  \
                                    rval[i:,:,i*dim:] + \
                                    inps[i:,:,begin*dim:end*dim])
        return rval

    def apply(self, state_below, state_below_struct, mask_below, init_state=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
        else:
            raise NotImplementedError

        # input to the gates, concatenated
        sb  = T.dot(state_below_struct, self.W_xs)
        ssb = T.dot(state_below_struct, self.W_xss)

        structures = self.shift_combine(sb) + self.b_xs
        structures_score = self.shift_combine(ssb) + self.b_xss

        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(self._step_forward_with_context,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            init_state1 = []
            init_state2 = []
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                init_state1 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                init_state2 = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            rval, updates = theano.scan(self._step_forward,
                                        sequences=[state_below, mask_below, structures, structures_score],
                                        outputs_info=[init_state, init_state1, init_state2],
                                        n_steps=n_steps
                                        )
        self.output = rval
        return self.output

    def merge_out(self, state_below, state_below_lstm, state_below_struct, mask_below, context=None):
        hiddens, memo_cells, structures = self.apply(state_below_lstm, state_below_struct, mask_below, context=context)
        if context is None:
            n_emb_all = self.n_emb_lstm + self.n_emb_struct - self.n_emb_share
            msize = n_emb_all + self.n_hids + self.n_shids # for structure
            osize = self.n_hids
            combine = T.concatenate([state_below, hiddens, structures], axis=2)
        else:
            msize = self.n_in + self.n_hids + self.c_hids
            osize = self.n_hids
            n_times = state_below.shape[0]
            m_context = repeat_x(context, n_times)
            combine = T.concatenate([state_below, hiddens, m_context], axis=2)

        self.W_m = param_init().normal((msize, osize*2), 'W_m')
        self.b_m = param_init().constant((osize*2,), 'b_m')
        self.params += [self.W_m, self.b_m]

        merge_out = theano.dot(combine, self.W_m) + self.b_m
        merge_max = merge_out.reshape((merge_out.shape[0],
                                       merge_out.shape[1],
                                       merge_out.shape[2]/2,
                                       2), ndim=4).max(axis=3)
        return merge_max * mask_below[:, :, None]

def _slice(_x, n, dim):
    if   _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    elif _x.ndim == 2:
        return _x[:, n*dim:(n+1)*dim]
    elif _x.ndim == 1:
        return _x[n*dim:(n+1)*dim]
    else :
        raise NotImplementedError

class rnn_pyramid_layer(object):

    def __init__(self, n_in, n_in_emb, n_hids, with_contex=False, **kwargs):
        self.n_in = n_in
        self.n_in_emb = n_in_emb
        self.n_hids = n_hids
        self.with_contex = with_contex
        if self.with_contex:
            self.c_hids = kwargs.pop('c_hids', n_hids)
        self._init_params()

    def _init_params(self):
        n_in = self.n_in
        n_in_emb = self.n_in_emb
        n_hids = self.n_hids
        size_xh = (n_in, n_hids)
        size_eh = (n_in_emb, n_hids)
        size_hh = (n_hids, n_hids)
        self.W_xi = param_init().normal(size_xh)
        self.W_xf = param_init().normal(size_xh)
        self.W_xo = param_init().normal(size_xh)
        self.W_xc = param_init().normal(size_xh)

        self.W_hi = param_init().orth(size_hh)
        self.W_hf = param_init().orth(size_hh)
        self.W_ho = param_init().orth(size_hh)
        self.W_hc = param_init().orth(size_hh)

        self.b_i = param_init().constant((n_hids,))
        self.b_f = param_init().constant((n_hids,), scale=1.0) #  Jozefowicz et al. 2015
        self.b_o = param_init().constant((n_hids,))
        self.b_c = param_init().constant((n_hids,))

        self.params = [self.W_xi, self.W_xf, self.W_xo, self.W_xc,
                       self.W_hi, self.W_hf, self.W_ho, self.W_hc,
                       self.b_i, self.b_f, self.b_o, self.b_c ]

        if False: # use rnn pyramid
            self.W_ep = param_init().normal(size_eh)
            self.W_hp = param_init().orth(size_hh)
            self.b_p = param_init().constant((n_hids,))

            self.params = self.params \
                        + [self.W_ep, self.W_hp, self.b_p]


        if True: # use attetion step
            self.W_hg = param_init().orth(size_hh)

            self.W_pc = param_init().orth(size_hh)
            self.b_pc = param_init().constant((n_hids,))

            self.params = self.params \
                        + [self.W_hg, self.W_pc, self.b_pc]

        #if False: # use gru pyramid
        self.gru_z = True
        if not self.gru_z :
            self.W_grup_ru = param_init().multi_orth((n_hids,2*n_hids))
            self.W_grup_hh = param_init().orth(size_hh)
            self.W_grup_x  = param_init().normal((n_in_emb, 2*n_hids))
            self.W_grup_xx = param_init().normal(size_eh)
            self.b_grup_x  = param_init().constant((2*n_hids,))
            self.b_grup_xx = param_init().constant((n_hids,))
            self.params = self.params \
                        + [self.W_grup_ru, self.W_grup_hh,
                           self.W_grup_x, self.W_grup_xx,
                           self.b_grup_x, self.b_grup_xx]
        else:
            self.gru_pyramid = GRU_Z(n_in_emb, n_hids)
            self.params.extend(self.gru_pyramid.params)

        if True: #
            self.W_sz = param_init().normal(size_hh)
            self.W_sr = param_init().normal(size_hh)
            self.W_sh = param_init().normal(size_hh)

            self.W_hz = param_init().orth(size_hh)
            self.W_hr = param_init().orth(size_hh)
            self.W_hh = param_init().orth(size_hh)

            self.b_z = param_init().constant((n_hids,))
            self.b_r = param_init().constant((n_hids,))
            self.b_h = param_init().constant((n_hids,))

            if True:
                self.struct_begin = None
                #self.struct_begin = param_init().constant((n_hids,), scale=0.) # for struct begin tag <struct>
                #self.params = self.params + [self.struct_begin]
            if False:
                self.struct_end = param_init().constant((n_hids,), scale=0.) # for struct end tag <struct> after W_ep
                self.params = self.params + [self.struct_end]

            self.U_att = param_init().normal((n_hids, 1)) # for attention
            self.c_att = param_init().constant((1,), scale=1.0) # for attention

            self.params = self.params \
                        + [self.W_sz, self.W_sr, self.W_sh,
                           self.W_hz, self.W_hr, self.W_hh,
                           self.b_z, self.b_r, self.b_h,
                           self.U_att, self.c_att]

        if self.with_contex:
            size_ch = (self.c_hids, self.n_hids)
            self.W_cz = param_init().uniform(size_ch)
            self.W_cr = param_init().uniform(size_ch)
            self.W_ch = param_init().uniform(size_ch)
            self.W_c_init = param_init().uniform(size_ch)

            self.params = self.params + [self.W_cz, self.W_cr,
                                         self.W_ch, self.W_c_init]

    def _rnn_step(self, x, m, h_tm1, W, t, n, W_x=None, b_x=None, shifted_begin_tag=None):
        '''
        x: input gating at time t
        m_t: mask at time t
        h_tm1: hidden state at time t (output of step, could not be computed outside)
        W: W for h
        W_x, b_x : W, b for x (if not pre computed outside this function)
        '''
        pre_x = x
        if W_x != None: # if not pre computed outside, compute here
            assert b_x != None
            pre_x = T.dot(x, W_x) + b_x

        if shifted_begin_tag:
            h_tm1 = T.set_subtensor(h_tm1[:,t,:], shifted_begin_tag[None,:])

        preact = T.dot(h_tm1, W)
        if preact.ndim == 2:
            preact += pre_x
            m_t = m[:, None]
        elif preact.ndim == 3:
            #preact += pre_x[:, None, :] # for multiple rnn share same W,b
            if T.lt(t+1, n):
                preact = T.set_subtensor(preact[:,:t+1], preact[:,:t+1] + pre_x[:, None, :])
            else:
                raise ValueError("idx out of bound t:{} n:{}".format(t,n))
            m_t = m[:, None, None]
        else:
            raise NotImplementedError

        h_t = T.tanh(preact)

        h_t = m_t * h_t + (1. - m_t) * h_tm1
        return h_t

    def _gru_pyramid_step(self, n_hid, x, xx, m, h_tm1, W_ru, W_o, t, n, W_x=None, W_xx=None, b_x=None, b_xx=None, shifted_begin_tag=None):
        '''
        x: input gating at time t
        xx: input at time t
        m_t: mask at time t
        h_tm1: hidden state at time t (output of step, could not be computed outside)
        W_r|u|d: W for h
        W_[x|xx][r|u|d] : W for x (if not pre computed outside this function)

        call with pre computed x and xx _gru_step(n_hid, x, xx, m_t, h_tm1, W_ru, W_o)
        '''
        pre_x = x
        pre_xx = xx
        if W_x != None: # if not pre computed outside, compute here
            assert W_xx != None and b_x != None and b_xx != None
            pre_x = T.dot(x, W_x) + b_x
            pre_xx = T.dot(x, W_xx) + b_xx

        if shifted_begin_tag:
            h_tm1 = T.set_subtensor(h_tm1[:,t,:], shifted_begin_tag[None,:])

        preact = T.dot(h_tm1, W_ru)
        preactx = T.dot(h_tm1, W_o)
        if preact.ndim == 2:
            assert preactx.ndim == 2 and pre_x.ndim == 1 and pre_xx.ndim == 1
            preact = T.inc_subtensor(preact[:t+1], pre_x[None, :])
            m_t = m[:, None]
        elif preact.ndim == 3:
            assert preactx.ndim == 3
            assert pre_x.ndim == 2
            assert pre_xx.ndim == 2
            if T.lt(t+1, n):
                preact = T.inc_subtensor(preact[:,:t+1], pre_x[:, None, :])
            else:
                raise ValueError("idx out of bound t:{} n:{}".format(t,n))
            m_t = m[:, None, None]
        else:
            raise NotImplementedError

        preact = T.nnet.sigmoid(preact)

        r = _slice(preact, 0, n_hid)
        u = _slice(preact, 1, n_hid)

        preactx *= r
        if preact.ndim == 2:
            preactx = T.inc_subtensor(preactx[:t+1], pre_xx[None, :])
        elif preact.ndim == 3:
            preactx = T.inc_subtensor(preactx[:,:t+1], pre_xx[:, None, :])
        else:
            raise NotImplementedError

        h_t = T.tanh(preactx)

        h_t = u * h_tm1 + (1. - u) * h_t
        h_t = m_t * h_t + (1. - m_t) * h_tm1
        return h_t

    def _gru_pyramid_step_member(self, n_hid, x, xx, m, t, h_tm1, n, shifted_begin_tag):
        '''
        x: input gating at time t
        xx: input at time t
        m_t: mask at time t
        h_tm1: hidden state at time t (output of step, could not be computed outside)
        W_r|u|d: W for h
        W_[x|xx][r|u|d] : W for x (if not pre computed outside this function)

        call with pre computed x and xx _gru_step(n_hid, x, xx, m_t, h_tm1, W_ru, W_o)
        '''
        pre_x = x
        pre_xx = xx
        if self.W_grup_x != None: # if not pre computed outside, compute here
            pre_x = T.dot(x, self.W_grup_x) + self.b_grup_x
            pre_xx = T.dot(x, self.W_grup_xx) + self.b_grup_xx

        if shifted_begin_tag:
            h_tm1 = T.set_subtensor(h_tm1[:,t,:], shifted_begin_tag[None,:])

        preact = T.dot(h_tm1, self.W_grup_ru)
        preactx = T.dot(h_tm1, self.W_grup_hh)
        if preact.ndim == 2:
            assert preactx.ndim == 2 and pre_x.ndim == 1 and pre_xx.ndim == 1
            preact = T.inc_subtensor(preact[:t+1], pre_x[None, :])
            m_t = m[:, None]
        elif preact.ndim == 3:
            assert preactx.ndim == 3
            assert pre_x.ndim == 2
            assert pre_xx.ndim == 2
            if T.lt(t+1, n):
                preact = T.inc_subtensor(preact[:,:t+1], pre_x[:, None, :])
            else:
                raise ValueError("idx out of bound t:{} n:{}".format(t,n))
            m_t = m[:, None, None]
        else:
            raise NotImplementedError

        preact = T.nnet.sigmoid(preact)

        r = _slice(preact, 0, n_hid)
        u = _slice(preact, 1, n_hid)

        preactx *= r
        if preact.ndim == 2:
            preactx = T.inc_subtensor(preactx[:t+1], pre_xx[None, :])
        elif preact.ndim == 3:
            preactx = T.inc_subtensor(preactx[:,:t+1], pre_xx[:, None, :])
        else:
            raise NotImplementedError

        h_t = T.tanh(preactx)

        h_t = u * h_tm1 + (1. - u) * h_t
        h_t = m_t * h_t + (1. - m_t) * h_tm1
        return h_t

    def _gru_step(self, n_hid, x, xx, m_t, h_tm1, W_ru, W_o, W_x=None, W_xx=None, b_x=None, b_xx=None):
        '''
        x: input gating at time t
        xx: input at time t
        m_t: mask at time t
        h_tm1: hidden state at time t (output of step, could not be computed outside)
        W_r|u|d: W for h
        W_[x|xx][r|u|d] : W for x (if not pre computed outside this function)

        call with pre computed x and xx _gru_step(n_hid, x, xx, m_t, h_tm1, W_ru, W_o)
        '''
        pre_x = x
        pre_xx = xx
        if W_x != None: # if not pre computed outside, compute here
            assert W_xx != None and b_x != None and b_xx != None
            pre_x = T.dot(x, W_x) + b_x
            pre_xx = T.dot(x, W_xx) + b_xx

        preact = T.dot(h_tm1, W_ru)
        preact += pre_x
        preact = T.nnet.sigmoid(preact)

        r = _slice(preact, 0, n_hid)
        u = _slice(preact, 1, n_hid)

        preactx = T.dot(h_tm1, W_o)
        preactx *= r
        preactx += pre_xx

        h_t = T.tanh(preactx)

        h_t = u * h_tm1 + (1. - u) * h_t
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_tm1
        return h_t


    def _lstm_step(self, n_hid, x, m_t, h_tm1, c_tm1, W, W_x=None, b_x=None):
        '''
        x: input gating at time t
        m_t: mask at time t
        h_tm1: hidden state at time t (output of step, could not be computed outside)
        W: W for h
        W_x, b_x : W, b for x (if not pre computed outside this function)

        call with pre computed x and xx _gru_step(n_hid, x, xx, m_t, h_tm1, W_ru, W_o)
        '''
        pre_x = x
        if W_x != None: # if not pre computed outside, compute here
            assert b_x != None
            pre_x = T.dot(x, W_x) + b_x

        preact = T.dot(h_tm1, W)
        preact += pre_x

        i = _slice(preact, 0, n_hid)
        f = _slice(preact, 1, n_hid)
        o = _slice(preact, 2, n_hid)
        c = _slice(preact, 3, n_hid)

        i = T.nnet.sigmoid(i)
        f = T.nnet.sigmoid(f)
        o = T.nnet.sigmoid(o)
        c = T.tanh(c)

        c = f * c_tm1 + i * c
        c_t = m_t[:, None] * c + (1. - m_t)[:, None] * c_tm1

        h_t = o * T.tanh(c)
        h_t = m_t[:, None] * h_t + (1. - m_t)[:, None] * h_tm1
        return h_t, c_t


    def _attention_step(self, h, pre_ctx, ctx, ctx_mask, W_comb_att, U_att, c_att, W_x=None, b_x=None):
        '''
        x: input gating at time t
        m_t: mask at time t
        h_tm1: hidden state at time t (output of step, could not be computed outside)
        W: W for h
        W_x, b_x : W, b for ctx (if not pre computed outside this function)
        '''
        if W_x != None: # if not pre computed outside, compute here
            assert b_x != None
            pre_ctx = T.dot(ctx, W_x) + b_x

        pstate = T.dot(h, W_comb_att)
        pctx = pre_ctx + pstate[None, :, :]
        pctx = T.tanh(pctx)

        alpha = T.dot(pctx, U_att) + c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)

        if ctx_mask:
            alpha = alpha * ctx_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_psum = (ctx * alpha[:, :, None]).sum(0)  # current context

        return ctx_psum, alpha.T


    def _pyramid_step_lazy(self, x_t, x_m, t, h_tm1, n_steps):
        '''
        when x_h/z/r are not calculated outside
        '''
        x_h =  T.dot(x_t, self.W_grup_xx) + self.b_grup_xx
        x_zr = T.dot(x_t, self.W_grup_x) + self.b_grup_x
        return self._pyramid_step(x_h, x_zr, x_m, t, h_tm1, n_steps)

    def _pyramid_step(self, x_h, x_zr, x_m, t, h_tm1, n_steps):
        '''
        x_h/z/r: input at time t     shape=[batch, hid] or [hid]
        x_m: mask of x_t         shape=[batch] or [1]
        h_tm1: previous state    shape=[batch, t+1 or n_steps, hid] or [t+1 or n_steps, hid]
        '''
        if False:
            if x_h.ndim == 1 and h_tm1.ndim == 2:
                h_tm1 = T.set_subtensor(h_tm1[t,:], self.struct_begin_tag)
            elif x_h.ndim == 2 and h_tm1.ndim == 3:
                h_tm1 = T.set_subtensor(h_tm1[:,t,:], self.struct_begin_tag[None,:])
            else:
                raise NotImplementedError

        zr_t = T.dot(h_tm1, self.W_grup_ru)
        can_h_t = T.dot(h_tm1, self.W_grup_hh)

        if x_h.ndim == 1 and h_tm1.ndim == 2:
            xzr = x_zr[None,:]
            xm = x_m[:,None]

            zr_t = T.inc_subtensor(zr_t[:t+1], xzr)
        elif x_h.ndim == 2 and h_tm1.ndim == 3:
            xzr = x_zr[:,None,:]
            xm = x_m[:,None,None]

            zr_t = T.inc_subtensor(zr_t[:,:t+1], xzr)
        else:
            raise NotImplementedError

        zr_t = T.nnet.sigmoid(zr_t)

        z_t = _slice(zr_t, 0, self.n_hids)
        r_t = _slice(zr_t, 1, self.n_hids)

        can_h_t *= r_t

        if x_h.ndim == 1 and h_tm1.ndim == 2:
            xh = x_h[None,:]
            can_h_t = T.inc_subtensor(can_h_t[:t+1], xh)
        elif x_h.ndim == 2 and h_tm1.ndim == 3:
            xh = x_h[:,None,:]
            can_h_t = T.inc_subtensor(can_h_t[:,:t+1], xh)
        else:
            raise NotImplementedError

        can_h_t = T.tanh(can_h_t)

        h_t = z_t * h_tm1 + (1 - z_t) * can_h_t
        h_t = xm * h_t + (1. - xm) * h_tm1

        return h_t

    def _step_forward(self, x_t, e_t, x_m, t, h_tm1, c_tm1, s_tm1, p_tm1, n_steps):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        c_tm1: previous memory cell
        '''
        W = T.concatenate([self.W_hi, self.W_hf, self.W_ho, self.W_hc], axis=1)
        h1, c1 = self._lstm_step(self.n_hids, x_t, x_m, h_tm1, c_tm1, W)

        h_t = h1
        c_t = c1

        #p_t = self._rnn_step(e_t, x_m, p_tm1, self.W_hp, t, self.n_steps, shifted_begin_tag=self.struct_begin)
        #chunks_t = self._rnn_step(self.struct_end_x, x_m, p_t, self.W_hp, t, self.n_steps)

        if self.gru_z :
            p_t = self.gru_pyramid._pyramid_step_lazy(e_t, x_m, t, p_tm1)
            #p_t = self.gru_pyramid._gru_pyramid_step_lazy(e_t, x_m, t, p_tm1, self.n_steps)
        elif False:
            p_t = self._gru_pyramid_step(self.n_hids, e_t, e_t, x_m, p_tm1, \
                                         self.gru_pyramid.W_hzr, self.gru_pyramid.W_hh, \
                                         t, n_steps, \
                                         self.gru_pyramid.W_xzr, self.gru_pyramid.W_xh, \
                                         self.gru_pyramid.b_zr, self.gru_pyramid.b_h, \
                                         shifted_begin_tag=None)
        elif True:
            p_t = self._gru_pyramid_step_member(n_hid=self.n_hids,
                                                x=e_t,
                                                xx=e_t,
                                                m=x_m,
                                                t=t,
                                                h_tm1=p_tm1,
                                                n=self.n_steps,
                                                shifted_begin_tag=self.struct_begin)
        elif False:
            p_t = self._pyramid_step_lazy(e_t, x_m, t, p_tm1, self.n_steps)
        else:
            p_t = self._gru_pyramid_step(self.n_hids, e_t, e_t, x_m, p_tm1, \
                                         self.W_grup_ru, self.W_grup_hh, \
                                         t, self.n_steps, \
                                         self.W_grup_x, self.W_grup_xx, \
                                         self.b_grup_x, self.b_grup_xx, \
                                         shifted_begin_tag=self.struct_begin)
        chunks_t = p_t

        ctx = chunks_t.dimshuffle(1, 0, 2)
        pre_ctx = T.dot(ctx, self.W_pc) + self.b_pc

        s_t, alphaT = self._attention_step(h_t, pre_ctx, ctx, None, self.W_hg, self.U_att, self.c_att)

        W_ru = T.concatenate([self.W_hz, self.W_hr], axis=1)
        W_s_ru = T.concatenate([self.W_sz, self.W_sr], axis=1)
        b_s_ru = T.concatenate([self.b_z, self.b_r], axis=0)
        h_t = self._gru_step(self.n_hids, s_t, s_t, x_m, h_t, W_ru, self.W_hh, W_s_ru, self.W_sh, b_s_ru, self.b_h)

        return h_t, c_t, s_t, p_t

    def apply(self, state_below, emb_below, mask_below, init_state_h=None, context=None):
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
            n_steps = state_below.shape[0]
            self.n_steps = n_steps
        else:
            raise NotImplementedError

        W_x = T.concatenate([self.W_xi, self.W_xf, self.W_xo, self.W_xc], axis=1)
        b_x = T.concatenate([self.b_i, self.b_f, self.b_o, self.b_c], axis=0)
        state_below_x = T.dot(state_below, W_x) + b_x
        emb_below_x = emb_below #T.dot(emb_below, self.W_ep) + self.b_p
        if False:
            self.struct_end_x = self.struct_end[None,:] * T.ones_like(emb_below_x[0])

        if self.with_contex:
            if init_state is None:
                init_state = T.tanh(theano.dot(context, self.W_c_init))
            c_z = theano.dot(context, self.W_cz)
            c_r = theano.dot(context, self.W_cr)
            c_h = theano.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]
            rval, updates = theano.scan(self._step_forward_with_context,
                                        sequences=[state_below, mask_below],
                                        outputs_info=[init_state],
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        else:
            if init_state_h is None:
                init_state_h = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                init_state_c = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                init_state_s = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
                init_rnn_h = T.alloc(numpy.float32(0.), batch_size, n_steps, self.n_hids)
                #init_rnn_h = T.set_subtensor(init_rnn_h[:,0,:], self.struct_begin[None,:])
            #non_sequences = [chunk_begin_state]
            #rnn_pyramid_h = T.alloc(numpy.float32(0.), n_steps, batch_size, self.n_hids)
            #lstm_h = T.alloc(numpy.float32(0.), n_steps, batch_size, self.n_hids)
            #lstm_c = T.alloc(numpy.float32(0.), n_steps, batch_size, self.n_hids)
            #lstm_s = T.alloc(numpy.float32(0.), n_steps, batch_size, self.n_hids)
            step_idx =  T.arange(n_steps)

            outputs_info=[init_state_h, init_state_c, init_state_s, init_rnn_h]
            non_sequences = [n_steps]
            rval, updates = theano.scan(self._step_forward,
                                        sequences=[state_below_x, emb_below_x, mask_below, step_idx],
                                        outputs_info=outputs_info,
                                        non_sequences=non_sequences,
                                        n_steps=n_steps
                                        )

        self.output = rval
        return self.output

class lookup_table(object):
    def __init__(self, embsize, vocab_size, name='Wemb'):
        self.W = param_init().uniform((vocab_size, embsize), name)
        self.params = [self.W]
        self.vocab_size = vocab_size
        self.embsize = embsize

    def apply(self, indices, emb_idxs):
        outshape = [indices.shape[i] for i in range(indices.ndim)] \
                 + [emb_idxs.shape[0]]

        word_idxs = indices.flatten()
        return self.W[word_idxs[:,None], emb_idxs].reshape(outshape)
        #return self.W[numpy.ix_(indices.flatten(), emb_idxs)].reshape(outshape)

def dropout_layer(state_before, use_noise, dropout=0.5, trng=RandomStreams(1234)):
    proj = T.switch(use_noise,
                    state_before * trng.binomial(state_before.shape, p=(1.0 - dropout), n=1,
                                                 dtype=state_before.dtype),
                    state_before * (1. - dropout))
    return proj

class maxout_layer(object):
        def apply(self, state_below, n_in, n_out, mask, n_fold=2):
            msize = n_in #state_below.shape[2]
            osize = n_out

            self.W_m = param_init().normal((msize, osize*n_fold), 'W_m')
            self.b_m = param_init().constant((osize*n_fold,), 'b_m')
            self.params = [self.W_m, self.b_m]

            merge_out = theano.dot(state_below, self.W_m) + self.b_m
            merge_max = merge_out.reshape((merge_out.shape[0],
                                           merge_out.shape[1],
                                           merge_out.shape[2]/n_fold,
                                           n_fold), ndim=4).max(axis=3)
            return merge_max * mask[:, :, None]

class auto_encoder(object):
    def __init__(self, sentence, sentence_mask, vocab_size, n_in, n_hids, **kwargs):
        layers = []

        batch_size = sentence.shape[1]
        encoder = GRU(n_in, n_hids, with_contex=False)
        layers.append(encoder)

        if 'table' in kwargs:
            table = kwargs['table']
        else:
            table = lookup_table(n_in, vocab_size)
        # layers.append(table)

        state_below = table.apply(sentence)
        context = encoder.apply(state_below, sentence_mask)[-1]

        decoder = GRU(n_in, n_hids, with_contex=True)
        layers.append(decoder)

        decoder_state_below = table.apply(sentence[:-1])
        hiddens = decoder.merge_out(decoder_state_below,
                                    sentence_mask[:-1], context=context)

        logistic_layer = LogisticRegression(hiddens, n_hids, vocab_size)
        layers.append(logistic_layer)

        self.cost = logistic_layer.cost(sentence[1:],
                                        sentence_mask[1:])/batch_size
        self.output = context
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)









