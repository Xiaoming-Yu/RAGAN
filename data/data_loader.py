from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .textDataset import TextDataset


def CreateDataLoader(opt):
    if opt.c_type == 'text' or opt.c_type == 'image_text':
      split = 'train' if opt.isTrain else 'test'
      use_gray = True if opt.model == 'supervised' else False
      dataset = TextDataset(opt.dataroot,
                            split=split,
                            load_size=opt.loadSize,
                            fine_size=opt.fineSize,
                            is_flip = not opt.no_flip,
                            use_gray=use_gray)
      opt.n_words = dataset.n_words
    else:
      raise NotImplementedError('c_type [%s] is not found' % opt.c_type)
    
        
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=opt.batchSize,
                             shuffle=opt.isTrain,
                             drop_last=True,
                             num_workers=opt.nThreads)
                             
    return data_loader