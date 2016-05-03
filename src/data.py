#encoding=utf-8
import configurations
import pprint

path = "../data/uy_origin/"
out_path = "../data/uy_data2/"
def process(filename):
    with open(path + filename , "r") as fin:
        fin_word = open(out_path + filename + "_word" , "w")
        fin_char = open(out_path + filename + "_morph" , "w")
        fin_rel = open(out_path + filename + "_rel" , "w")
        for line in fin.readlines():
            tmps = line.strip().split(' ')
            l1 = ""
            l2 = ""
            rel = []
            for tmp in tmps:
                tmp2 = tmp.split('/')
                l1 += tmp2[0] + " "
                l2 += tmp2[1].replace('+' , ' ') + " "
                idx = len(tmp2[1].split("+"))
                rel.append(idx)
            fin_word.write(l1.strip() + "\n")
            fin_char.write(l2.strip() + "\n")
            fin_rel.write(" ".join([str(a) for a in rel]) + "\n")
        fin_rel.close()
        fin_word.close()
        fin_char.close()

if __name__ == "__main__":
    #configuration = getattr(configurations , 'get_config_penn')()
    #print pprint.pformat(configuration)
    process("train")
    process("test")
    process("valid")
