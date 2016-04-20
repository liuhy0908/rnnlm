import theano
import theano.tensor as T
import numpy
import logging
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from itertools import izip

logger = logging.getLogger(__name__)

class param_init(object):

    def __init__(self,**kwargs):

        self.shared = kwargs.pop('shared', True)

    def param(self, size, name="param", init_type='uniform', **kwargs):
        try:
            func = getattr(self, init_type)
        except AttributeError:
            logger.error('AttributeError, {}'.format(init_type))
        else:
            param = func(size, **kwargs)
        if self.shared:
            param = theano.shared(value=param, name=name, borrow=True)
        return param

    def uniform(self, size, name="uniform", **kwargs):
        low = kwargs.pop('low', -6./sum(size))
        high = kwargs.pop('high', 6./sum(size))
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.uniform(low=low, high=high, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, name=name, borrow=True)
        return param

    def normal(self, size, name="normal", **kwargs):
        loc = kwargs.pop('loc', 0.)
        scale = kwargs.pop('scale', 0.05)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        param = numpy.asarray(
            rng.normal(loc=loc, scale=scale, size=size),
            dtype=theano.config.floatX)
        if self.shared:
            param = theano.shared(value=param, name=name, borrow=True)
        return param

    def constant(self, size, name="constant", **kwargs):
        value = kwargs.pop('scale', 0.)
        param = numpy.ones(size, dtype=theano.config.floatX)*value
        if self.shared:
            param = theano.shared(value=param, name=name, borrow=True)
        return param

    def _qr(self, dim, **kwargs):
        scale = kwargs.pop('scale', 1.)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        if dim < 0:
            raise ValueError
        M = rng.randn(dim,dim).astype(theano.config.floatX)
        Q, R = numpy.linalg.qr(M)
        Q = Q * numpy.sign(numpy.diag(R))
        param = Q*scale
        return param

    def orth(self, size, name="orth", **kwargs):
        scale = kwargs.pop('scale', 1.)
        rng = kwargs.pop('rng', numpy.random.RandomState(1234))
        if len(size) != 2:
            raise ValueError
        if size[0] == size[1]:
            param = self._qr(size[0])
            if self.shared:
                param = theano.shared(value=param, name=name, borrow=True)
            return param
        else:
            M1 = rng.randn(size[0], size[0]).astype(theano.config.floatX)
            M2 = rng.randn(size[1], size[1]).astype(theano.config.floatX)
            Q1, R1 = numpy.linalg.qr(M1)
            Q2, R2 = numpy.linalg.qr(M2)
            Q1 = Q1 * numpy.sign(numpy.diag(R1))
            Q2 = Q2 * numpy.sign(numpy.diag(R2))
            n_min = min(size[0], size[1])
            param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale
            if self.shared:
                param = theano.shared(value=param, name=name, borrow=True)
            return param

    def multi_orth(self, size, name="multi_orth", **kwargs):
        if numpy.mod(size[0],size[1]) or numpy.mod(size[1],size[0]):
            n_max = max(size[0], size[1])
            n_min = min(size[0], size[1])
            l_axis=0
            if size[0] < size[1]:
                l_axis=1
            param = numpy.concatenate([self._qr(n_min) for i in range(n_max/n_min)], axis=l_axis)
        else:
            raise ValueError
        if self.shared:
            param = theano.shared(value=param, name=name, borrow=True)
            return param

def  repeat_x(x, n_times):
    # This is black magic based on broadcasting,
    # that's why variable names don't make any sense.
    a = T.shape_padleft(x)
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    out = a * b
    return out

def adadelta(parameters, gradients, hard_clip=15, rho=0.95, eps=1e-6):
    # create variables to store intermediate updates
    gradients_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32')) for p in parameters ]
    deltas_sq = [theano.shared(numpy.zeros(p.get_value().shape, dtype='float32')) for p in parameters ]

    # calculates the new "average" delta for the next iteration
    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

    # calculates the step in direction. The square root is an approximation to getting the RMS for the average value
    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    # calculates the new "average" deltas for the next step.
    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    # Prepare it as a list f
    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)

    if hard_clip > 0:
        parameters_updates = [ (p,T.clip(p - d, -hard_clip,hard_clip)) for p,d in izip(parameters,deltas) ]
    else:
        parameters_updates = [ (p,(p - d)) for p,d in izip(parameters,deltas) ]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates


def step_clipping(params, gparams, scale=1., nan_scale=1, skip_nan_batch=1):
    nan_num = theano.shared(numpy.zeros([1], dtype='float32'))
    inf_num = theano.shared(numpy.zeros([1], dtype='float32'))
    #skip = theano.shared(numpy.zeros([1,1], dtype='float32'))

    nan_num = sum(map(lambda x: T.isnan(x).sum(), gparams))
    inf_num = sum(map(lambda x: T.isinf(x).sum(), gparams))
    notfinite = nan_num + inf_num
    grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
    #notfinite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    skip = T.switch(skip_nan_batch, notfinite, 0)
    multiplier = T.switch(grad_norm < scale, 1., scale / grad_norm)
    _g = []
    for param, gparam in izip(params, gparams):
        tmp_g = T.switch(scale > 0., gparam * multiplier, gparam) # FIXME, combine with multiplier logic
        _g.append(T.switch(skip > 0, param * (1 - nan_scale), tmp_g))

    params_clipping = _g

    return  params_clipping, nan_num, inf_num

# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)

def ReplicateLayer(x, n_times):
    a = T.shape_padleft(x)
    padding = [1] * x.ndim
    b = T.alloc(numpy.float32(1), n_times, *padding)
    return a * b

def concatenate(tensor_list, axis=0):
    concat_size = sum(tt.shape[axis] for tt in tensor_list)
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)
    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)
        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]
    return out
