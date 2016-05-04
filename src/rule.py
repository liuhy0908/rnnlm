import theano
import theano.tensor as T

config = {}
config['data_path'] = "../data/origin/"
config['train'] = config['data_path'] + "train_joint.txt"
config['test'] = config['data_path'] + "test.txt"
config['test_ans'] = config['data_path'] + "test_in.ref"
config['test_in'] = config['data_path'] + "test_in.txt"
def calc(_path):
    fin = open(_path , "r")
    cnt_num = 0
    cnt_diff = 0
    wdict = {}
    stem_dict = {}
    suffix_dict = {}
    for line in fin:
        words = line.strip().split()
        for word in words:
            tmps = word.split('/')
            if '+' in tmps[1]:
                stems = tmps[1].split('+')
                for i , stem in enumerate(stems):
                    if i == 0:
                        stem_dict[stem] = [1]
                    else:
                        suffix_dict[stem] = [1]
                if tmps[0] not in wdict:
                    wdict[tmps[0]] = [tmps[1]]
                    cnt_num += 1
                    if tmps[0] != tmps[1].replace('+' , ''):
                        cnt_diff += 1
                else:
                    if tmps[1] not in wdict[tmps[0]]:
                        wdict[tmps[0]].append(tmps[1])
                        cnt_num += 1
                        if tmps[0] != tmps[1].replace('+' , ''):
                            cnt_diff += 1
    print "cnt_num : " , cnt_num , "cnt_diff : " , cnt_diff ,
    print "ratio : " , 1.0 * cnt_diff / cnt_num
    fin.close()
    return wdict , stem_dict , suffix_dict
def merge():
    fin = open(config['test_in'] , "r")
    fin2 = open(config['test_ans'] , "r")
    fout = open(config['test'] , "w")
    lines1 = fin.readlines()
    lines2 = fin2.readlines()
    for line1 , line2 in zip(lines1 , lines2):
        tmps = line1.strip().split()
        tmps2 = line2.strip().split()
        if len(tmps) != len(tmps2):
            print 'word not align'
        line = ""
        for word1 , word2 in zip(tmps , tmps2):
            line += word1 + "/" + word2 + " "
        fout.write(line + "\n")
    fin.close()
    fin2.close()
    fout.close()

def output(train_dict , test_dict):
    cnt = 0
    cnt2 = 0
    for key in test_dict:
        if key not in train_dict:
            cnt += 1
        else:
            tmps = test_dict[key]
            for tmp in tmps:
                if tmp not in train_dict[key]:
                    cnt2 += 1
    print 'key not hit num: ' , cnt
    print 'value not hit num: ', cnt2
    print 'test_dict size: ' , len(test_dict)

def main():
    train_dict , train_stems , train_suffix = calc(config['train'])
    test_dict , test_stems , test_suffix = calc(config['test'])
    output(train_dict , test_dict)
    output(train_stems , test_stems)
    output(train_suffix , test_suffix)

if __name__ == "__main__":
    main()
