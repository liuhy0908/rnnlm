# basic unit: RNN, GRU, LSTM, ATTENTION, FeedForward, FF2, FF3
import numpy
import theano
import theano.tensor as T
from initialization import constant_weight, uniform_weight, ortho_weight, multi_orth, norm_weight
from theano.tensor.nnet import categorical_crossentropy
from mfd_utils import ReplicateLayer, _p, concatenate
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from numpy.random import RandomState as RandomStreams # FIXME: make it right


def _slice(_x, n, dim):
    if   _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    elif _x.ndim == 2:
        return _x[:, n*dim:(n+1)*dim]
    elif _x.ndim == 1:
        return _x[n*dim:(n+1)*dim]
    else :
        raise NotImplementedError

class ATTENTION(object):

    def __init__(self, n_hids, rng=RandomStreams(1234), name='ATTENTION'):

        self.n_hids = n_hids
        self.rng = rng
        self.pname = name
        self.params = []
        self._init_params()

    def _init_params(self):

        shape_hh = (self.n_hids, self.n_hids)

        self.W_comb_att = norm_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_comb_att'))
        self.U_att = norm_weight(rng=self.rng, shape=(self.n_hids, 1), name=_p(self.pname, 'U_att'))
        self.c_att = constant_weight(shape=(1,), name=_p(self.pname, 'c_att'))
        self.params = [self.W_comb_att, self.U_att, self.c_att]

    def apply(self, h, pre_ctx, ctx, ctx_mask=None):
        '''
        h: h.shape = [batch, hid]
        pre_ctx:  shape = [src_len, batch, hid]
        ctx:  shape = [src_len, batch, ctx_hid]
        '''
        pstate = T.dot(h, self.W_comb_att)
        pctx = pre_ctx
        pctx += pstate[None, :, :]
        pctx = T.tanh(pctx)

        alpha = T.dot(pctx, self.U_att) + self.c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = T.exp(alpha)

        if ctx_mask:
            alpha = alpha * ctx_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_psum = (ctx * alpha[:, :, None]).sum(0)  # current context

        return ctx_psum, alpha.T

        
    def apply_3d_h(self, h, pre_ctx, ctx, ctx_mask=None):
        '''
        h: h.shape = [batch, trg_len, hid]
        pre_ctx:  shape = [src_len, batch, hid]
        ctx:  shape = [src_len, batch, ctx_hid]
        '''
        pstate = T.dot(h, self.W_comb_att)
        pctx = pre_ctx[:, :, None, :]
        pctx += pstate[None, :, :, :]
        pctx = T.tanh(pctx) # [src_len, batch, trg_len, hid]

        alpha = T.dot(pctx, self.U_att) + self.c_att
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1], alpha.shape[2]])
        alpha = T.exp(alpha)# [src_len, batch, trg_len]

        if ctx_mask:
            alpha = alpha * ctx_mask[:,:,None]
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_psum = (ctx[:,:,None,:] * alpha[:,:,:,None]).sum(0)  # current context
        # ctx_psum.shape = [batch, trg_len, ctx_hid]
        return ctx_psum, alpha.T

class FeedForward(object):

    def __init__(self, n_in, n_out, rng=RandomStreams(1234), orth=False, name='FeedForward'):

        self.n_in = n_in
        self.n_out = n_out
        self.orth = orth
        self.rng = rng
        self.pname = name
        self.params = []
        self._init_params()

    def _init_params(self):

        shape_io = (self.n_in, self.n_out)

        if self.orth:
            if self.n_in != self.n_out :
                raise ValueError('n_in != n_out when require orth in FeedForward')
            self.W = ortho_weight(rng=self.rng, shape=shape_io, name=_p(self.pname, 'W'))
        else:
            self.W = norm_weight(rng=self.rng, shape=shape_io, name=_p(self.pname, 'W'))
        self.b = constant_weight(shape=(self.n_out, ), name=_p(self.pname, 'b'))
        self.params = [self.W, self.b]

    def apply(self, state_below, activ='lambda x: theano.tensor.tanh(x)'):
        return eval(activ)(T.dot(state_below, self.W) + self.b)

class FF2_1(object):

    def __init__(self, n_in_0, n_in_1, n_out, rng=RandomStreams(1234), orth=False, name='FeedForward'):

        self.n_in_0 = n_in_0
        self.n_in_1 = n_in_1
        self.n_out = n_out
        self.orth = orth
        self.rng = rng
        self.pname = name
        self.params = []
        self._init_params()

    def _init_params(self):

        shape_io = (self.n_in_0, self.n_out)

        if self.orth:
            if self.n_in_0 != self.n_out: 
                raise ValueError('n_in != n_out when require orth in FeedForward')
            self.W = ortho_weight(rng=self.rng, shape=shape_io, name=_p(self.pname, 'W'))
        else:
            self.W = norm_weight(rng=self.rng, shape=shape_io, name=_p(self.pname, 'W'))
        self.params = [self.W]

        self.ff =  FeedForward(self.n_in_1, self.n_out, orth=self.orth, rng=self.rng, name=_p(self.pname, 'FF_W') )
        self.params.extend(self.ff.params)

    def apply(self, state_below_0, state_below_1, activ='lambda x: theano.tensor.tanh(x)'): 
        return eval(activ)(T.dot(state_below, self.W) + ff.apply(state_below_1, activ='lambda x: x')) #FIXME: eval() arg1 must be string?

class FF2(object):

    def __init__(self, n_in_0, n_in_1, n_out, rng=RandomStreams(1234), orth=False, name='FeedForward'):

        self.n_in_0 = n_in_0
        self.n_in_1 = n_in_1
        self.n_out = n_out
        self.orth = orth
        self.rng = rng
        self.pname = name
        self.params = []
        self._init_params()

    def _init_params(self):

        shape_i0o = (self.n_in_0, self.n_out)
        shape_i1o = (self.n_in_1, self.n_out)

        if self.orth:
            if self.n_in_0 != self.n_out or self.n_in_1 != self.n_out:
                raise ValueError('n_in != n_out when require orth in FeedForward')
            self.W0 = ortho_weight(rng=self.rng, shape=shape_i0o, name=_p(self.pname, 'W0'))
            self.W1 = ortho_weight(rng=self.rng, shape=shape_i1o, name=_p(self.pname, 'W1'))
        else:
            self.W0 = norm_weight(rng=self.rng, shape=shape_i0o, name=_p(self.pname, 'W0'))
            self.W1 = norm_weight(rng=self.rng, shape=shape_i1o, name=_p(self.pname, 'W1'))
        self.b = constant_weight(shape=(self.n_out, ), name=_p(self.pname, 'b'))
        self.params = [self.W0, self.W1, self.b]

    #def apply(self, state_below0, state_below1, activ='lambda x: theano.tensor.tanh(x)'):
    #    rval = T.dot(state_below0, self.W0) + T.dot(state_below1, self.W1) + self.b
    #    return eval(activ)(rval)
    def apply(self, state_below0, state_below1):
        rval = T.dot(state_below0, self.W0) + T.dot(state_below1, self.W1) + self.b
        return T.tanh(rval)

class FF3(object):

    def __init__(self, n_in_0, n_in_1, n_in_2, n_out, rng=RandomStreams(1234), orth=False, name='FeedForward'):

        self.n_in_0 = n_in_0
        self.n_in_1 = n_in_1
        self.n_in_2 = n_in_2
        self.n_out = n_out
        self.orth = orth
        self.rng = rng
        self.pname = name
        self.params = []
        self._init_params()

    def _init_params(self):

        shape_i0o = (self.n_in_0, self.n_out)
        shape_i1o = (self.n_in_1, self.n_out)
        shape_i2o = (self.n_in_2, self.n_out)

        if self.orth:
            if self.n_in_0 != self.n_out or self.n_in_1 != self.n_out or self.n_in_2 != self.n_out:
                raise ValueError('n_in != n_out when require orth in FeedForward')
            self.W0 = ortho_weight(rng=self.rng, shape=shape_i0o, name=_p(self.pname, 'W0'))
            self.W1 = ortho_weight(rng=self.rng, shape=shape_i1o, name=_p(self.pname, 'W1'))
            self.W2 = ortho_weight(rng=self.rng, shape=shape_i2o, name=_p(self.pname, 'W2'))
        else:
            self.W0 = norm_weight(rng=self.rng, shape=shape_i0o, name=_p(self.pname, 'W0'))
            self.W1 = norm_weight(rng=self.rng, shape=shape_i1o, name=_p(self.pname, 'W1'))
            self.W2 = norm_weight(rng=self.rng, shape=shape_i2o, name=_p(self.pname, 'W2'))
        self.b = constant_weight(shape=(self.n_out, ), name=_p(self.pname, 'b'))
        self.params = [self.W0, self.W1, self.W2, self.b]

    def apply(self, state_below0, state_below1, state_below2, activ='lambda x: theano.tensor.tanh(x)'):
        return eval(activ)(T.dot(state_below0, self.W0) + T.dot(state_below1, self.W1) + T.dot(state_below2, self.W2) + self.b)

class GRU(object):

    def __init__(self, n_in, n_hids, n_att_ctx=None, rng=RandomStreams(1234), name='GRU', with_context=False, with_begin_tag=False, with_end_tag=False):

        self.n_in = n_in
        self.n_hids = n_hids
        self.rng = rng
        self.n_att_ctx = n_att_ctx
        self.pname = name

        self.with_context = with_context
        self.with_begin_tag = with_begin_tag
        self.with_end_tag = with_end_tag
        if self.with_context:
            self.c_hids = n_hids

        self.params = []
        self._init_params()


    def _init_params(self):

        shape_xh = (self.n_in, self.n_hids)
        shape_xh2 = (self.n_in, 2*self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)
        shape_hh2 = (self.n_hids, 2*self.n_hids)

        self.W_xzr = norm_weight(rng=self.rng, shape=shape_xh2, name=_p(self.pname, 'W_xzr'))
        self.W_xh  = norm_weight(rng=self.rng, shape=shape_xh, name=_p(self.pname, 'W_xh'))
        self.b_zr  = constant_weight(shape=(2*self.n_hids, ), name=_p(self.pname, 'b_zr'))
        self.b_h   = constant_weight(shape=(self.n_hids, ), name=_p(self.pname, 'b_h'))
        self.W_hzr = multi_orth(rng=self.rng, size=shape_hh2, name=_p(self.pname, 'W_hzr'))
        self.W_hh  = ortho_weight(rng=self.rng, shape=shape_hh, name=_p(self.pname, 'W_hh'))

        self.params += [self.W_xzr, self.W_xh,
                        self.W_hzr, self.W_hh,
                        self.b_zr,  self.b_h]
        
        if self.with_context:
            shape_ch = (self.c_hids, self.n_hids)
            self.W_cz = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cz'))
            self.W_cr = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cr'))
            self.W_ch = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_ch'))
            self.W_c_init = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_c_init'))

            self.params += [self.W_cz, self.W_cr, self.W_ch, self.W_c_init]
        
        if self.with_begin_tag:
            self.struct_begin_tag = constant_weight(shape=(self.n_hids,), value=0., name=_p(self.pname, 'struct_begin_tag')) 
            self.params += [self.struct_begin_tag]

        if self.with_end_tag:
            self.struct_end_tag = constant_weight(shape=(self.n_in,), value=0., name=_p(self.pname, 'struct_end_tag')) 
            self.params += [self.struct_end_tag]

        if self.n_att_ctx:
            self.gru_combine_ctx_h = GRU(self.n_att_ctx, self.n_hids, rng=self.rng, name=_p(self.pname, 'gru_combine_ctx_h'))
            self.params.extend(self.gru_combine_ctx_h.params)
            self.attention = ATTENTION(self.n_hids, self.rng, name=_p(self.pname, 'att_ctx')) 
            self.params.extend(self.attention.params)

    def _pyramid_attention_step_lazy(self, x_t, x_m, t, h_tm1, pre_ctx, ctx, ctx_mask=None):
        '''
        when x_h/z/r are not calculated outside
        '''
        x_h = T.dot(x_t, self.W_xh) + self.b_h
        x_zr = T.dot(x_t, self.W_xzr) + self.b_zr
        return self._pyramid_attention_step(x_h, x_zr, x_m, t, h_tm1, pre_ctx, ctx, ctx_mask)

    def _pyramid_attention_step(self, x_h, x_zr, x_m, t, h_tm1, pre_ctx, ctx, ctx_mask=None):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        xm = x_m[:, None, None]
        h_1 = self._pyramid_step(x_h, x_zr, x_m, t, h_tm1)
        h_1 = xm * h_1 + (1. - xm)*h_tm1

        ctx_psum, alphaT = self.attention.apply_3d_h(h_1, pre_ctx, ctx, ctx_mask)

        h_2 = self.gru_combine_ctx_h._step_lazy(ctx_psum, x_m, h_1)
        h_2 = xm * h_2 + (1. - xm)*h_tm1

        h_t = h_2
        return h_t


    def _pyramid_step_lazy(self, x_t, x_m, t, h_tm1):
        '''
        when x_h/z/r are not calculated outside
        '''
        x_h =  T.dot(x_t, self.W_xh) + self.b_h
        x_zr = T.dot(x_t, self.W_xzr) + self.b_zr
        return self._pyramid_step(x_h, x_zr, x_m, t, h_tm1)

    def _pyramid_step(self, x_h, x_zr, x_m, t, h_tm1):
        '''
        x_h/z/r: input at time t     shape=[batch, hid] or [hid]
        x_m: mask of x_t         shape=[batch] or [1]
        h_tm1: previous state    shape=[batch, t+1 or n_steps, hid] or [t+1 or n_steps, hid]
        '''
        if self.with_begin_tag:
            if x_h.ndim == 1 and h_tm1.ndim == 2:
                h_tm1 = T.set_subtensor(h_tm1[t,:], self.struct_begin_tag)
            elif x_h.ndim == 2 and h_tm1.ndim == 3:
                h_tm1 = T.set_subtensor(h_tm1[:,t,:], self.struct_begin_tag[None,:])
            else:
                raise NotImplementedError

        zr_t = T.dot(h_tm1, self.W_hzr)
        can_h_t = T.dot(h_tm1, self.W_hh)

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


    def _attention_step(self, x_h, x_zr, x_m, h_tm1, pre_ctx, ctx, ctx_mask=None):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        xm = x_m[:, None]
        h_1 = self._step(x_h, x_zr, x_m, h_tm1)
        h_1 = xm * h_1 + (1. - xm)*h_tm1

        ctx_psum, alphaT = self.attention.apply(h_1, pre_ctx, ctx, ctx_mask)

        h_2 = self.gru_combine_ctx_h._step_lazy(ctx_psum, x_m, h_1)
        h_2 = xm * h_2 + (1. - xm)*h_tm1

        h_t = h_2
        return h_t


    def _step_lazy(self, x_t, x_m, h_tm1):
        '''
        when x_h/z/r are not calculated outside
        '''
        x_h = T.dot(x_t, self.W_xh) + self.b_h
        x_zr = T.dot(x_t, self.W_xzr) + self.b_zr
        return self._step(x_h, x_zr, x_m, h_tm1)

    def _step(self, x_h, x_zr, x_m, h_tm1):
        '''
        deal with 1d or 2d x_h/zr
        x_h/zr: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        if x_h.ndim == 1 :
            xm = x_m
        elif x_h.ndim == 2 :
            xm = x_m[:,None] 
        elif x_h.ndim == 3 :
            xm = x_m[:,None,None]
        elif x_h.ndim == 4 :
            xm = x_m[:,None,None,None]
        else:
            raise NotImplementedError

        zr_t = T.nnet.sigmoid(x_zr + T.dot(h_tm1, self.W_hzr))

        z_t = _slice(zr_t, 0, self.n_hids)
        r_t = _slice(zr_t, 1, self.n_hids)

        can_h_t = T.tanh(x_h + r_t * T.dot(h_tm1, self.W_hh))

        h_t = z_t * h_tm1 + (1. - z_t) * can_h_t
        h_t = xm * h_t + (1. - xm) * h_tm1
        return h_t


    def _step_context(self, x_t, x_m, h_tm1, cz, cr, ch):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        zr_t = T.nnet.sigmoid(T.dot(x_t, self.W_xzr) +
                             T.dot(h_tm1, self.W_hzr) +
                             T.dot(cz, self.W_czr) + self.b_zr) # FIXME: make czr

        z_t = _slice(zr_t, 0, self.n_hids)
        r_t = _slice(zr_t, 1, self.n_hids)

        can_h_t = T.tanh(T.dot(x_t, self.W_xh) +
                         r_t * T.dot(h_tm1, self.W_hh) +
                         T.dot(ch, self.W_ch) + self.b_h)

        h_t = z_t * h_tm1 + (1 - z_t) * can_h_t
        h_t = x_m[:, None] * h_t + (1. - x_m[:, None])*h_tm1
        return h_t


    def apply_pyramid(self, state_below, mask_below=None, init_state=None, context=None):
        '''
        state_below: shape=[seq_len, batch, n_in]
        init_state:  shape=[batch, seq_len, hid]
        '''
        n_steps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
            state_below = state_below.reshape((n_steps, batch_size, state_below.shape[1]))

        if mask_below is None:
            mask_below = T.alloc(numpy.float32(1.), n_steps, 1)

        if self.with_context:
            assert context

            if init_state is None:
                init_state = T.tanh(T.dot(context, self.W_c_init))

            c_z = T.dot(context, self.W_cz)
            c_r = T.dot(context, self.W_cr)
            c_h = T.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]

            rval, updates = theano.scan(self._step_context,
                                        sequences=[state_below, mask_below],
                                        non_sequences=non_sequences,
                                        outputs_info=[init_state],
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, n_steps, self.n_hids)

            state_below_xh = T.dot(state_below, self.W_xh) + self.b_h
            state_below_xzr = T.dot(state_below, self.W_xzr) + self.b_zr
            step_idx =  T.arange(n_steps)
            sequences = [state_below_xh, state_below_xzr, mask_below, step_idx]
            outputs_info=[init_state]
            non_sequences = []
            rval, updates = theano.scan(self._pyramid_step,
                                        sequences=sequences,
                                        outputs_info=outputs_info,
                                        non_sequences=non_sequences,
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        self.output = rval
        return self.output


    def apply(self, state_below, mask_below=None, init_state=None, context=None):

        n_steps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
            state_below = state_below.reshape((n_steps, batch_size, state_below.shape[1]))

        if mask_below is None:
            mask_below = T.alloc(numpy.float32(1.), n_steps, 1)

        if self.with_context:
            assert context

            if init_state is None:
                init_state = T.tanh(T.dot(context, self.W_c_init))

            c_z = T.dot(context, self.W_cz)
            c_r = T.dot(context, self.W_cr)
            c_h = T.dot(context, self.W_ch)
            non_sequences = [c_z, c_r, c_h]

            rval, updates = theano.scan(self._step_context,
                                        sequences=[state_below, mask_below],
                                        non_sequences=non_sequences,
                                        outputs_info=[init_state],
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        else:
            if init_state is None:
                init_state = T.alloc(numpy.float32(0.), batch_size, self.n_hids)

            state_below_xh = T.dot(state_below, self.W_xh) + self.b_h
            state_below_xzr = T.dot(state_below, self.W_xzr) + self.b_z
            sequences = [state_below_xh, state_below_xzr, mask_below]
            rval, updates = theano.scan(self._step,
                                        sequences=sequences,
                                        outputs_info=[init_state],
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        self.output = rval
        return self.output


class LSTM(object):

    def __init__(self, n_in, n_hids, n_att_ctx=None, seq_pyramid=False, rng=RandomStreams(1234), name='LSTM', with_context=False, with_begin_tag=False, with_end_tag=False):

        self.n_in = n_in
        self.n_hids = n_hids
        self.rng = rng
        self.n_att_ctx = n_att_ctx
        self.seq_pyramid = seq_pyramid 
        self.pname = name

        self.with_context = with_context
        self.with_begin_tag = with_begin_tag
        self.with_end_tag = with_end_tag
        if self.with_context:
            self.c_hids = n_hids

        self.params = []
        self._init_params()


    def _init_params(self):

        shape_xh = (self.n_in, self.n_hids)
        shape_xh4 = (self.n_in, 4*self.n_hids)
        shape_hh = (self.n_hids, self.n_hids)
        shape_hh4 = (self.n_hids, 4*self.n_hids)

        self.W_pre_x = norm_weight(rng=self.rng, shape=shape_xh4, name=_p(self.pname, 'W_pre_x'))
        self.W_h = multi_orth(rng=self.rng, size=shape_hh4, name=_p(self.pname, 'W_h'))

        b_i = constant_weight(share=False, shape=(self.n_hids, ), name=_p(self.pname, 'b_i'))
        b_f = constant_weight(share=False, value=1., shape=(self.n_hids, ), name=_p(self.pname, 'b_f'))
        b_o = constant_weight(share=False, shape=(self.n_hids, ), name=_p(self.pname, 'b_o'))
        b_c = constant_weight(share=False, shape=(self.n_hids, ), name=_p(self.pname, 'b_c'))
        b_ifoc = numpy.concatenate([b_i, b_f, b_o, b_c], axis=0)
        self.b_pre_x = theano.shared(value=b_ifoc, borrow=True, name=_p(self.pname, 'b_pre_x'))

        self.params += [self.W_pre_x, self.W_h, self.b_pre_x]
        
        if self.with_context:
            raise NotImplementedError
            #shape_ch = (self.c_hids, self.n_hids)
            #self.W_cz = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cz'))
            #self.W_cr = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_cr'))
            #self.W_ch = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_ch'))
            #self.W_c_init = norm_weight(rng=self.rng, shape=shape_ch, name=_p(self.pname, 'W_c_init'))

            #self.params += [self.W_cz, self.W_cr, self.W_ch, self.W_c_init]
        
        if self.with_begin_tag:
            self.struct_begin_tag = constant_weight(shape=(self.n_hids,), value=0., name=_p(self.pname, 'struct_begin_tag')) 
            self.params += [self.struct_begin_tag]

        if self.with_end_tag:
            self.struct_end_tag = constant_weight(shape=(self.n_in,), value=0., name=_p(self.pname, 'struct_end_tag')) 
            self.params += [self.struct_end_tag]

        if self.n_att_ctx:
            self.lstm_combine_ctx_h = LSTM(self.n_att_ctx, self.n_hids, rng=self.rng, name=_p(self.pname, 'lstm_combine_ctx_h'))
            self.params.extend(self.lstm_combine_ctx_h.params)
            self.attention = ATTENTION(self.n_hids, self.rng, name=_p(self.pname, 'att_ctx')) 
            self.params.extend(self.attention.params)
            if self.seq_pyramid:
                self.pyramid_on_seq = LSTM(self.n_att_ctx, self.n_att_ctx, rng=self.rng, name=_p(self.pname, 'pyramid_on_seq'))
                self.params.extend(self.pyramid_on_seq.params)
                self.ff_pyramid2ctx = FeedForward(self.n_att_ctx, self.n_hids, name=_p(self.pname, 'ff_pyramid2ctx'))
                self.params.extend(self.ff_pyramid2ctx.params)

    def _pyramid_attention_step_lazy(self, x_t, x_m, t, h_tm1, c_tm1, pre_ctx, ctx, ctx_mask=None):
        '''
        when x_h/z/r are not calculated outside
        '''
        pre_x_t = T.dot(x_t, self.W_pre_x) + self.b_pre_x
        return self._pyramid_attention_step(pre_x_t, x_m, t, h_tm1, c_tm1, pre_ctx, ctx, ctx_mask)

    def _pyramid_attention_step(self, pre_x_t, x_m, t, h_tm1, c_tm1, pre_ctx, ctx, ctx_mask=None):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        xm = x_m[:, None, None]
        h_1, c_1 = self._pyramid_step(pre_x_t, x_m, t, h_tm1, c_tm1)
        h_1 = xm * h_1 + (1. - xm)*h_tm1
        c_1 = xm * c_1 + (1. - xm)*c_tm1

        ctx_psum, alphaT = self.attention.apply_3d_h(h_1, pre_ctx, ctx, ctx_mask)

        h_2, c_2 = self.lstm_combine_ctx_h._step_lazy(ctx_psum, x_m, h_1, c_1)
        h_2 = xm * h_2 + (1. - xm)*h_tm1
        c_2 = xm * c_2 + (1. - xm)*c_tm1

        h_t = h_2
        c_t = c_2

        return h_t, c_t


    def _seq_pyramid_step(self, pre_x_t, pre_xp_t, x_m, t, h_tm1, c_tm1, hp_tm1, cp_tm1):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        xm = x_m[:, None]
        h_1, c_1 = self._step(pre_x_t, x_m, h_tm1, c_tm1)
        h_1 = xm * h_1 + (1. - xm)*h_tm1
        c_1 = xm * c_1 + (1. - xm)*c_tm1

        xmp = x_m[:, None, None]
        hp_1, cp_1 = self.pyramid_on_seq._pyramid_step(pre_xp_t, x_m, t, hp_tm1, cp_tm1)
        hp_t = xmp * hp_1 + (1. - xmp)*hp_tm1
        cp_t = xmp * cp_1 + (1. - xmp)*cp_tm1

        ctx = hp_t.dimshuffle(1, 0, 2)[:t+1]
        pre_ctx = self.ff_pyramid2ctx.apply(ctx)

        ctx_psum, alphaT = self.attention.apply(h_1, pre_ctx, ctx)

        h_2, c_2 = self.lstm_combine_ctx_h._step_lazy(ctx_psum, x_m, h_1, c_1)
        h_2 = xm * h_2 + (1. - xm)*h_tm1
        c_2 = xm * c_2 + (1. - xm)*c_tm1

        h_t = h_2
        c_t = c_2

        return h_t, c_t, hp_t, cp_t

    def _pyramid_step_lazy(self, x_t, x_m, t, h_tm1, c_tm1):
        '''
        when x_h/z/r are not calculated outside
        '''
        pre_x_t =  T.dot(x_t, self.W_pre_x) + self.b_pre_x
        return self._pyramid_step(pre_x_t, x_m, t, h_tm1, c_tm1)

    def _pyramid_step(self, pre_x_t, x_m, t, h_tm1, c_tm1):
        '''
        x_h/z/r: input at time t     shape=[batch, hid] or [hid]
        x_m: mask of x_t         shape=[batch] or [1]
        h_tm1: previous state    shape=[batch, t+1 or n_steps, hid] or [t+1 or n_steps, hid]
        '''
        if self.with_begin_tag:
            if pre_x_t.ndim == 1 and h_tm1.ndim == 2:
                h_tm1 = T.set_subtensor(h_tm1[t,:], self.struct_begin_tag)
            elif pre_x_t.ndim == 2 and h_tm1.ndim == 3:
                h_tm1 = T.set_subtensor(h_tm1[:,t,:], self.struct_begin_tag[None,:])
            else:
                raise NotImplementedError

        tmp_sum = T.dot(h_tm1, self.W_h)

        if pre_x_t.ndim == 1 and h_tm1.ndim == 2:
            xm = x_m[:,None]
            pre_x_t = pre_x_t[None,:]
            tmp_sum = T.inc_subtensor(tmp_sum[:t+1], pre_x_t)
        elif pre_x_t.ndim == 2 and h_tm1.ndim == 3:
            xm = x_m[:,None,None]
            pre_x_t = pre_x_t[:,None,:]
            tmp_sum = T.inc_subtensor(tmp_sum[:,:t+1], pre_x_t)
        else:
            raise NotImplementedError

        i = T.nnet.sigmoid(_slice(tmp_sum, 0, self.n_hids))
        f = T.nnet.sigmoid(_slice(tmp_sum, 1, self.n_hids))
        o = T.nnet.sigmoid(_slice(tmp_sum, 2, self.n_hids))
        c = T.tanh(        _slice(tmp_sum, 3, self.n_hids))

        c = f * c_tm1 + i * c
        c = xm * c + (1. - xm) * c_tm1

        h_t = o * T.tanh(c)

        h_t = xm * h_t + (1. - xm) * h_tm1

        return h_t, c


    def _attention_step(self, pre_x_t, x_m, h_tm1, c_tm1, pre_ctx, ctx, ctx_mask=None):
        '''
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        xm = x_m[:, None]
        h_1, c_1 = self._step(pre_x_t, x_m, h_tm1, c_tm1)
        h_1 = xm * h_1 + (1. - xm)*h_tm1
        c_1 = xm * c_1 + (1. - xm)*c_tm1

        ctx_psum, alphaT = self.attention.apply(h_1, pre_ctx, ctx, ctx_mask)

        h_2, c_2 = self.lstm_combine_ctx_h._step_lazy(ctx_psum, x_m, h_1, c_1)
        h_2 = xm * h_2 + (1. - xm)*h_tm1
        c_2 = xm * c_2 + (1. - xm)*c_tm1

        h_t = h_2
        c_t = c_2

        return h_t, c_2


    def _step_lazy(self, x_t, x_m, h_tm1, c_tm1):
        '''
        when x_h/z/r are not calculated outside
        '''
        pre_x_t =  T.dot(x_t, self.W_pre_x) + self.b_pre_x
        return self._step(pre_x_t, x_m, h_tm1, c_tm1)

    def _step(self, pre_x_t, x_m, h_tm1, c_tm1):
        '''
        deal with 1d or 2d x_h/zr
        x_h/zr: input at time t
        x_m: mask of x_t
        h_tm1: previous state
        '''
        if   pre_x_t.ndim == 1 :
            xm = x_m
        elif pre_x_t.ndim == 2 :
            xm = x_m[:,None] 
        elif pre_x_t.ndim == 3 :
            xm = x_m[:,None,None]
        elif pre_x_t.ndim == 4 :
            xm = x_m[:,None,None,None]
        else:
            raise NotImplementedError

        tmp_sum = T.dot(h_tm1, self.W_h)
        tmp_sum += pre_x_t

        i = T.nnet.sigmoid(_slice(tmp_sum, 0, self.n_hids))
        f = T.nnet.sigmoid(_slice(tmp_sum, 1, self.n_hids))
        o = T.nnet.sigmoid(_slice(tmp_sum, 2, self.n_hids))
        c = T.tanh(        _slice(tmp_sum, 3, self.n_hids))

        c = f * c_tm1 + i * c
        c = xm * c + (1. - xm) * c_tm1

        h_t = o * T.tanh(c)
        h_t = xm * h_t + (1. - xm) * h_tm1
        return h_t, c

    def apply_pyramid(self, state_below, 
                            mask_below=None, 
                            init_state_h=None, 
                            init_state_c=None, 
                            context=None):
        '''
        state_below: shape=[seq_len, batch, n_in]
        init_state:  shape=[batch, seq_len, hid]
        '''
        n_steps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
            state_below = state_below.reshape((n_steps, batch_size, state_below.shape[1]))

        if mask_below is None:
            mask_below = T.alloc(numpy.float32(1.), n_steps, 1)

        if self.with_context:
            raise NotImplementedError
        else:
            if init_state_h is None:
                init_state_h = T.alloc(numpy.float32(0.), batch_size, n_steps, self.n_hids)
            if init_state_c is None:
                init_state_c = T.alloc(numpy.float32(0.), batch_size, n_steps, self.n_hids)

            state_below_pre = T.dot(state_below, self.W_pre_x) + self.b_pre_x
            step_idx        = T.arange(n_steps)
            sequences       = [state_below_pre, mask_below, step_idx]
            outputs_info    = [init_state_h, init_state_c]
            non_sequences   = []

            rval, updates = theano.scan(self._pyramid_step,
                                        sequences=sequences,
                                        outputs_info=outputs_info,
                                        non_sequences=non_sequences,
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        self.output = rval
        return self.output


    def apply_seq_pyramid(self, state_below, 
                                state_below_p, 
                                mask_below=None, 
                                init_state_h=None, 
                                init_state_c=None, 
                                init_state_hp=None, 
                                init_state_cp=None, 
                                context=None):
        '''
        state_below: shape=[seq_len, batch, n_in]
        state_below_p: shape=[seq_len, batch, n_in_p]
        init_state:  shape=[batch, seq_len, hid]
        '''
        n_steps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
            state_below = state_below.reshape((n_steps, batch_size, state_below.shape[1]))
            state_below_p = state_below_p.reshape((n_steps, batch_size, state_below_p.shape[1]))

        if mask_below is None:
            mask_below = T.alloc(numpy.float32(1.), n_steps, 1)

        if self.with_context:
            raise NotImplementedError
        else:
            if init_state_h is None:
                init_state_h = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            if init_state_c is None:
                init_state_c = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            if init_state_hp is None:
                init_state_hp = T.alloc(numpy.float32(0.), batch_size, n_steps, self.n_hids)
            if init_state_cp is None:
                init_state_cp = T.alloc(numpy.float32(0.), batch_size, n_steps, self.n_hids)

            state_below_pre = T.dot(state_below, self.W_pre_x) + self.b_pre_x
            state_below_p_pre = T.dot(state_below_p, self.pyramid_on_seq.W_pre_x) + self.pyramid_on_seq.b_pre_x
            step_idx        = T.arange(n_steps)
            sequences       = [state_below_pre, state_below_p_pre, mask_below, step_idx]
            outputs_info    = [init_state_h, init_state_c, init_state_hp, init_state_cp]
            non_sequences   = []

            rval, updates = theano.scan(self._seq_pyramid_step,
                                        sequences=sequences,
                                        outputs_info=outputs_info,
                                        non_sequences=non_sequences,
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        self.output = rval
        return self.output


    def apply(self, state_below, mask_below=None, init_state_h=None, init_state_c=None, context=None):
        '''
        state_below: shape=[seq_len, batch, n_in]
        init_state:  shape=[batch, seq_len, hid]
        '''
        n_steps = state_below.shape[0]
        if state_below.ndim == 3:
            batch_size = state_below.shape[1]
        else:
            batch_size = 1
            state_below = state_below.reshape((n_steps, batch_size, state_below.shape[1]))

        if mask_below is None:
            mask_below = T.alloc(numpy.float32(1.), n_steps, 1)

        if self.with_context:
            raise NotImplementedError
        else:
            if init_state_h is None:
                init_state_h = T.alloc(numpy.float32(0.), batch_size, self.n_hids)
            if init_state_c is None:
                init_state_c = T.alloc(numpy.float32(0.), batch_size, self.n_hids)

            state_below_pre = T.dot(state_below, self.W_pre_x) + self.b_pre_x

            sequences       = [state_below_pre, mask_below]
            outputs_info    = [init_state_h, init_state_c]
            non_sequences   = []

            rval, updates = theano.scan(self._step,
                                        sequences=sequences,
                                        outputs_info=outputs_info,
                                        non_sequences=non_sequences,
                                        name=_p(self.pname, 'layers'),
                                        n_steps=n_steps)
        self.output = rval
        return self.output


