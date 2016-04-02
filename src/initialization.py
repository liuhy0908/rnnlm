# initializations
import logging
import numpy
import theano

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rng = numpy.random.RandomState(1234)


def constant_weight(shape, value=0., name=None, share=True):
    param = numpy.ones(shape, dtype=theano.config.floatX) * value
    if share:
        return theano.shared(value=param, borrow=True, name=name)
    else:
        return param


# uniform initialization for weights
def uniform_weight(rng, shape, name=None):
    low = -6./sum(shape)
    high = 6./sum(shape)
    param = numpy.asarray(rng.uniform(low=low, high=high, size=shape),
                          dtype=theano.config.floatX)

    return theano.shared(value=param, borrow=True, name=name)


def _qr(dim, **kwargs):
    scale = kwargs.pop('scale', 1.)
    rng = kwargs.pop('rng', numpy.random.RandomState(1234))
    if dim < 0:
        raise ValueError
    M = rng.randn(dim,dim).astype(theano.config.floatX)
    Q, R = numpy.linalg.qr(M)
    Q = Q * numpy.sign(numpy.diag(R))
    param = Q*scale
    return param

def multi_orth(size, name="multi_orth", **kwargs):
    scale = kwargs.pop('scale', 1.)
    rng = kwargs.pop('rng', numpy.random.RandomState(1234))
    if numpy.mod(size[0],size[1]) or numpy.mod(size[1],size[0]):
        n_max = max(size[0], size[1])
        n_min = min(size[0], size[1])
        l_axis=0
        if size[0] < size[1]:
            l_axis=1
        param = numpy.concatenate([_qr(dim=n_min, scale=scale, rng=rng) for i in range(n_max/n_min)], axis=l_axis)
    else:
        raise ValueError
    if True: #self.shared:
        param = theano.shared(value=param, name=name, borrow=True)
        return param

# orthogonal initialization for weights
def ortho_weight(rng, shape, scale=1., name=None):
    #W = numpy.random.randn(ndim, ndim)
    #u, s, v = numpy.linalg.svd(W)
    #param = u.astype(theano.config.floatX)
    #return theano.shared(value=param, borrow=True, name=name)

    if len(shape) != 2:
        raise ValueError

    if shape[0] == shape[1]:
        M = rng.randn(*shape).astype(theano.config.floatX)
        Q, R = numpy.linalg.qr(M)
        Q = Q * numpy.sign(numpy.diag(R))
        param = Q*scale
        return theano.shared(value=param, borrow=True, name=name)

    M1 = rng.randn(shape[0], shape[0]).astype(theano.config.floatX)
    M2 = rng.randn(shape[1], shape[1]).astype(theano.config.floatX)
    Q1, R1 = numpy.linalg.qr(M1)
    Q2, R2 = numpy.linalg.qr(M2)
    Q1 = Q1 * numpy.sign(numpy.diag(R1))
    Q2 = Q2 * numpy.sign(numpy.diag(R2))
    n_min = min(shape[0], shape[1])
    param = numpy.dot(Q1[:, :n_min], Q2[:n_min, :]) * scale

    return theano.shared(value=param, borrow=True, name=name)


# weight initializer, normal by default
def norm_weight(rng, shape, loc=0, scale=0.01, name=None):
    param = numpy.asarray(rng.normal(loc=loc, scale=scale, size=shape),
                          dtype=theano.config.floatX)

    return theano.shared(value=param, borrow=True, name=name)

