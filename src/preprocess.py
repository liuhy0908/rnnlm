import logging
logging.basicConfig(level=logging.INFO)
import os
from collections import Counter
import configurations

import cPickle as pickle
logger = logging.getLogger("preprocess")

class PrepareData:
    def __init__(self, train_file, train_dic,
                 bos_token, eos_token,
                 vocab_size=1000, unk_id=0,
                 unk_token='<unk>', seq_len=50, **kwargs):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.filename = train_file
        self.vocabsize = vocab_size
        self.unk_id = unk_id
        self.unk_token = unk_token
        self.seq_len = seq_len
        if kwargs['train_morph_file'] != "":
            self.train_char_file = kwargs['train_morph_file']
            dic = self._creat_dic(self.train_char_file , kwargs['seq_morph_len'] , kwargs['morph_size'])
            f = open(kwargs['train_morph_dic'], 'wb')
            pickle.dump(dic, f)
            f.close()

        dic = self._creat_dic(self.filename , self.seq_len , self.vocabsize)
        f = open(train_dic, 'wb')
        pickle.dump(dic, f)
        f.close()
        logger.info('dump train dict {} has been dumped'.format(train_dic))

    def _creat_dic(self , filename , seq_len , vocabsize):
        if os.path.isfile(filename):
            train = open(filename)
        else:
            logger.warning("file name {} not exist".format(filename))

        worddict={}
        worddict[self.unk_token] = self.unk_id

        counter = Counter()
        for line in train:
            line_vec = line.strip().split()
            if isinstance(seq_len, int):
                line_vec = line_vec[:seq_len]
            counter.update(line_vec)

        index = self.unk_id + 1
        if self.unk_token in counter:
            del counter[self.unk_token]
        print len(counter)
        # most_common : return top n
        for word, c in counter.most_common(vocabsize-3): #FIXME 2 or 3
            worddict[word] = index
            index += 1

        if self.bos_token:
            worddict[self.bos_token] = len(worddict)
        if self.eos_token:
            worddict[self.eos_token] = len(worddict)

        #assert len(worddict) == self.vocabsize

        return worddict

if __name__ =="__main__":
    configuration = getattr(configurations, 'get_config_morph')()
    prepare = PrepareData(**configuration)
