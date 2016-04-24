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
def step(in1 , in2 , in3):
    in3 = 10
    return in1

def test3():
    k = T.lscalar("k")
    A = T.lvector("A")
    b = T.lscalar("b")
    result , updates = theano.scan(fn = step,
                                    sequences = A ,
                                    outputs_info = k,
                                    non_sequences = b,
                                    n_steps = 5)
    final_result = result
    power = theano.function(inputs = [A , k , b] , outputs = [final_result , b])
    print power(range(10) , 0 , 0)


if __name__ == "__main__":
    test3()
