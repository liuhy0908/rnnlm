import logging
logging.basicConfig(level=logging.INFO)
from fuel.datasets import TextFile
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Padding, SortMapping, Unpack, Mapping
import cPickle as pickle
import numpy as np
import random
logger = logging.getLogger(__name__)

def _length(sentence):
    return len(sentence[0])
def padding(mat):
    mat2 = []
    for line in mat:
        tmp = line.strip().split()
        l = [1]
        l.extend(int(a) for a in tmp)
        l.append(1)
        mat2.append(l)
    max_length = max([len(a) for a in mat2])
    for t in mat2:
        t.extend([0] * (max_length - len(t)))
    m = np.array(mat2)
    return m

def padding2(tensor3):
    maxs = []
    for mat in tensor3:
        tmp = max([len(a) for a in mat])
        maxs.append(tmp)
    max_length = max(maxs)
    for mat in tensor3:
        for vec in mat:
            vec.extend([0] * (max_length - len(vec)))
    return tensor3

def DStream(datatype, config):
    if datatype=='train':
        filename = config['train_file']
        filename_morph = config['train_morph_file']
        filename_rel = config['train_rel_file']
    elif datatype == 'valid':
        filename = config['valid_file']
        filename_morph = config['valid_morph_file']
        filename_rel = config['valid_rel_file']
    elif datatype == 'test':
        filename = config['test_file']
        filename_morph = config['test_morph_file']
        filename_rel = config['test_rel_file']
    else:
        logger.error('wrong datatype, train, valid, or test')
    data = TextFile(files=[filename],
                    dictionary=pickle.load(open(config['train_dic'],'rb')),
                    unk_token=config['unk_token'],
                    level='word',
                    bos_token=config['bos_token'],
                    eos_token=config['eos_token'])
    data_morph = TextFile(files=[filename_morph],
                    dictionary=pickle.load(open(config['train_morph_dic'],'rb')),
                    unk_token=config['unk_token'],
                    level='word',
                    bos_token=config['bos_token'],
                    eos_token=config['eos_token'])
    data_stream = DataStream.default_stream(data)
    data_stream.sources = ('sentence',)
    data_morph_stream = DataStream.default_stream(data_morph)
    data_morph_stream.sources = ('sentence',)
    # organize data in batches and pad shorter sequences with zeros
    batch_size = config['batch_size']
    rels_stream = []
    with open(filename_rel , "r") as fin:
        lines = fin.readlines()
        i = 0
        while i < len(lines):
            if i + batch_size < len(lines):
                rels_stream.append(padding(lines[i : i + batch_size]))
                i = i + batch_size
            else:
                rels_stream.append(padding(lines[i : len(lines)]))
                i = i + batch_size
    data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(batch_size))
    data_stream = Padding(data_stream)

    data_morph_stream = Batch(data_morph_stream, iteration_scheme=ConstantScheme(batch_size))
    data_morph_stream = Padding(data_morph_stream)
    data_morph_tensor3 = []
    mask_morph_tensor3 = []
    #data_morph_stream : batch_num * batch * sentence
    #rels_stream : batch_num * batch * sentence
    #dta_morph_tensor3 : batch_num * batch * sentence * morph
    for data_morph_tuple , rel in zip(data_morph_stream.get_epoch_iterator() , rels_stream):
        data_morph , mask_morph = data_morph_tuple
        #data_morph : batch * sentence
        #rel : batch * sentence
        tmp = []
        tmp_mask = []
        for m , mask , r in zip(data_morph , mask_morph , rel):
            #m : sentence
            #r : sentence
            start = 0
            tmp2 = []
            tmp_mask2 = []
            for idx in r:
                tmp2.append(m[start:start+idx].tolist())
                tmp_mask2.append(mask[start:start+idx].tolist())
                #print m[start:start+idx]
                start = start + idx
            #print len(tmp)
            #print padding2(tmp2)
            tmp.append(tmp2)
            tmp_mask.append(tmp_mask2)
            #print len(tmp) , tmp
            #print m , r
            #print m.shape , r.shape
        #print padding2(tmp)
        data_morph_tensor3.append(np.array(padding2(tmp)))
        mask_morph_tensor3.append(np.array(padding2(tmp_mask) , dtype='float32'))
    return data_stream , data_morph_tensor3 , mask_morph_tensor3

if __name__ == "__main__":
    import configurations
    configuration = getattr(configurations, 'get_config_morph')()
    ds , ds_morph , ds_morph_mask = DStream(datatype='train', config=configuration)
    """
    fin = open("../data/uy_data2/train_dic.pkl" , "r")
    #fin = open("../data/uy_data2/train_morph_dic.pkl" , "r")
    dic = pickle.load(fin)
    fin.close()
    dic2 = {}
    for key in dic:
        dic2[dic[key]] = key
    """
    i = 0
    start = 0
    for data_tuple , data_morph , mask_morph in zip(ds.get_epoch_iterator() , ds_morph , ds_morph_mask):
        data , mask = data_tuple
        #print type(data) , type(data_morph)
        print data.shape , data_morph.shape , mask_morph.shape
        #print data , mask , data_morph , mask_morph
        #print data.dtype , mask.dtype , data_morph.dtype , mask_morph.dtype
        #print data[0] , data_morph[0]
        #print data[0].shape , data_morph[0].shape
        #print data, mask
        #print data[0], mask[0]
        """
        for key in data[1]:
            if key in dic2 and key != 0:
                print dic2[key]
        """
        #print data[1], mask[1]
        #print len(data) , len(data[0])
        #print start , start + len(data)
        start += len(data)
        i += 1
        if i == 30:
            exit(0)
