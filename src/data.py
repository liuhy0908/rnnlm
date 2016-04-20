#encoding=utf-8
import configurations
import pprint
class DataMod(object):
    def __init__(self , **kwargs):
        self.train = kwargs['train_file']
        self.train_dic = kwargs['train_dic']
        self.valid = kwargs['valid_file']
        self.test = kwargs['test_file']
        self.vocab_size = kwargs['vocab_size']



if __name__ == "__main__":
    configuration = getattr(configurations , 'get_config_penn')()
    print pprint.pformat(configuration)
    dm = DataMod(**configuration)
    print dm.train
