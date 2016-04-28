#encoding=utf8
import cPickle as pickle
import numpy
import theano
import theano.tensor as T
def test():
    fin = open("../data/train_dic.pkl" , "r")
    dic = pickle.load(fin)
    print type(dic)
    i = 0
    for key in dic:
        print key , dic[key]
        i += 1
        if i >= 100:
            break
    fin.close()

def test2():
    m = numpy.asarray([[1 , 2 , 3 , 4 , 5] , [5 , 6 , 7 , 8 , 9] , [10 , 11 , 12 , 13 , 14]])
    print m
    a  = numpy.asarray([[0] , [2]])
    b = a.flatten()
    print b
    print b[:,None]
    print m[b[:,None],[0 ,2]]
def step(in1 , in2):
    return T.arange(in1)
    #return in1

def test3():
    A = T.vector("A")
    k = T.vector("k")
    #k = T.scalar("k")
    result , updates = theano.scan(fn = step,
                                    sequences = A ,
                                    outputs_info = k,
                                    n_steps = 5)
    power = theano.function(inputs = [A , k] , outputs = [result])
    print power([3,3,3,3,3] , range(3))


if __name__ == "__main__":
    test3()
