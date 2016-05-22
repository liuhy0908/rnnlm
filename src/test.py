#encoding=utf8
import cPickle as pickle
import numpy as np
import theano
import theano.tensor as T
from model import MorphStruct
def test():
    fin = open("../data/uy_data2/train_dic.pkl" , "r")
    #fin = open("../data/uy_data2/train_morph_dic.pkl" , "r")
    dic = pickle.load(fin)
    dic2 = {}
    for key in dic:
        dic2[dic[key]] = key
    #print dic2[0]
    i = 0
    print len(dic)
    for key in dic:
        #print key , dic[key]
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

def test4():
    a = np.array([[[0 , 1 , 2 , 3] , [4 , 5 , 6 , 7] , [7 , 8 , 9 , 0]],
                  [[1 , 0 , 1 , 0] , [2 , 0 , 2 , 0] , [3 , 0 , 3 , 0]]])
    b = np.array([[[1 , 0 , 1 , 0] , [1 , 0 , 1 , 0] , [1 , 0 , 1 , 0]],
                  [[1 , 1 , 1 , 1] , [2 , 2 , 2 , 2] , [3 , 3 , 3 , 3]]])
    c = np.array([[9 , 0 , 9] , [8 , 0 , 8]])
    print a.shape
    print a[:0 , : , :]
    #print a.sum(0)
    #print a.sum(1)
    #print a * b
    print c[: , : , None]
    print c[None , : , :]
    print c.shape
    print a * c[: , : , None]
    print (a * c[: , : , None]).sum(0)


def test5():
    a = T.vector("a")
    b = T.scalar("b")
    c = T.matrix("c")
    out = T.concatenate([a[None,:] , c] , axis = 0)
    fn = theano.function(inputs=[a , c] , outputs=[out])
    print fn([1 , 2 , 3] , [[4 ,  5 , 6] , [7 , 8 , 9]])

def test6():
    A = T.tensor3("A")
    k = T.tensor3("k")
    k = A[:,1:3,:].sum(1)
    #k = T.scalar("k")
    fn = theano.function(inputs = [A] , outputs = [k])
    i = [[[1,1,1],[2,2,2],[3,3,3],[4,4,4]],[[3,3,3],[4,4,4],[5,5,5],[6,6,6]]]
    print i
    print fn(i)

def test7():
    morph = T.tensor3("morph")
    morph_mask = T.tensor3("mask")
    rel = T.matrix("rel")
    morphStruct = MorphStruct()
    morph_out = morphStruct.apply(morph , morph_mask , rel)
    fn = theano.function(inputs = [morph , morph_mask , rel] ,outputs = [morph_out] , on_unused_input='ignore')
    #rel : batch * sentence
    #state_below_morph : batch * sentence * n_emb_morph
    i = [
            [
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4] ,
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4] ,
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4]
            ],
            [
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4] ,
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4] ,
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4]
            ],
            [
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4] ,
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4] ,
                [1,1,1,1,1] , [2,2,2,2,2] , [3,3,3,3,3] , [4,4,4,4,4]
            ]
        ]
    m = i
    r = [[1,1,2,1,3,1,3,0,0,0] , [1,2,1,2,1,1,1,1,1,1],[3,1,1,5,1,1,0,0,0,0]]
    #res = fn(i , m , r)
    mat = np.array(i)
    print mat.shape
    print mat.sum(2)
    print mat.sum(1).shape

def test8():
    '''
    i = [
            [[1 , 2 ] , [3 , 4]] , [[5 , 6] , [7 , 8]],
            [[11 , 22 ] , [33 , 44]] , [[55 , 66] , [77 , 88]],
            [[111 , 222 ] , [333 , 444]] , [[555 , 666] , [777 , 888]],
        ]
    '''
    i = [
            [
                [
                    [1 , 1 , 1],
                    [2 , 2 , 2],
                    [3 , 3 , 3]
                ],
                [
                    [11 , 11 , 11],
                    [22 , 22 , 22],
                    [33 , 33 , 33]
                ]
            ],
            [
                [
                    [4 , 4 , 4],
                    [5 , 5 , 5],
                    [6 , 6 , 6]
                ],
                [
                    [44 , 44 , 44],
                    [55 , 55 , 55],
                    [66 , 66 , 66]
                ]
            ],
            [
                [
                    [7 , 7 , 71],
                    [8 , 8 , 8],
                    [9 , 9 , 9]
                ],
                [
                    [77 , 77 , 77],
                    [88 , 88 , 88],
                    [99 , 99 , 99]
                ]
            ]
        ]
    m = np.array(i)
    print m.shape
    print m.transpose((2 , 0 , 1 , 3)).flatten()
    m2 = m.transpose((2 , 0 , 1 , 3))
    m3 = m2.reshape([3 , 6 , 3])
    print m3
    print m3[-1,:,:]
    print m3[-1,:,:].reshape([3 , 2 , 3])
    #m2 = np.transpose(m , [1 , 0 , 2])
    #print m2.shape
    #print m2



if __name__ == "__main__":
    test8()
