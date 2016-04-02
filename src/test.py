#encoding=utf8
import cPickle as pickle
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


if __name__ == "__main__":
    test()
