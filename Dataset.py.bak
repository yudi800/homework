#pian
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import gc
import random
from torch.utils.data.sampler import BatchSampler,RandomSampler
import scipy.io as sio
class myRandomSampler(RandomSampler):
    def __init__(self, data_source, replacement=False, num_samples=None,seed = 20):
        super(myRandomSampler,self).__init__(data_source=data_source, replacement=replacement,num_samples= num_samples)
        self.seed = seed
    def __iter__(self):
        seed = (self.seed+1)%10000
        setup_seed(20)
        n = len(self.data_source)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

# preprocessing of data
def setup_seed(seed):#设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(20)
def makedata():
    # setup_seed(20)
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    # tra = [transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        normalize]
    # dataset1 = datasets.ImageFolder('data/trainimg/continuum', transform=transforms.Compose(tra))
    # sampler1 = myRandomSampler(dataset1)
    # dataloader1 = torch.utils.data.DataLoader(dataset1,batch_size=128,sampler=sampler1)
    # dataset2 = datasets.ImageFolder('data/trainimg/magnetogram', transform=transforms.Compose(tra))
    # sampler2 = myRandomSampler(dataset1)
    # dataloader2 = torch.utils.data.DataLoader(dataset2,batch_size=128,sampler=sampler2)
    # print(dataset1.imgs==dataset2.imgs)
    dataloder,dataloder3 = gettraindataloder(128)
    # dataloder2 = gettestdataloder(128)
    x1 = torch.Tensor([])
    x2 = torch.Tensor([])
    y = torch.tensor([],dtype =torch.int64)
    for i in range(3):
        for step, (batch_x, batch_y) in enumerate(zip(dataloder,dataloder3)):
            print(str(i)+'  '+str(step))
            print(batch_x[1]==batch_y[1])
        # for step, (batch_x, batch_y) in enumerate(dataloder2):
        #     print(str(i) + '  ' + str(step))
        #     print(batch_x[1] == batch_y[1])
    # sio.savemat('data/testdataend.mat',{'x':x1,'y':y})

    print(x1.size())
def gettraindataloder(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    dataset1 = datasets.ImageFolder('data/trainimg/continuum', transform=transforms.Compose(tra))
    sampler1 = myRandomSampler(dataset1)

    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, sampler=sampler1)
    dataset2 = datasets.ImageFolder('data/trainimg/magnetogram', transform=transforms.Compose(tra))
    sampler2 = myRandomSampler(dataset1)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, sampler=sampler2)
    return dataloader1, dataloader2

def gettestdataloder(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    tra = [transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           normalize]

    dataset1 = datasets.ImageFolder('data/testimg/continuum', transform=transforms.Compose(tra))
    sampler1 = myRandomSampler(dataset1)

    dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, sampler=sampler1)
    dataset2 = datasets.ImageFolder('data/testimg/magnetogram', transform=transforms.Compose(tra))
    sampler2 = myRandomSampler(dataset1)
    dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, sampler=sampler2)
    return zip(dataloader1, dataloader2)

# makedata()