#Data set imports
from __future__ import print_function
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import urllib2
import zipfile
import os
import os.path
from PIL import Image
import glob
import random

class ImageSet(Dataset):
    def __init__(self):
        self.url = "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/"
        self.dataPath = "./datasets/"
        self.loaded = 0
    
    def loadData(self, 
                 setName, 
                 mode, 
                 img_size): 
        
        self.transform = transforms.Compose([transforms.Resize(int(img_size*1.12), Image.BICUBIC),
                      transforms.RandomCrop(img_size),
                      transforms.RandomHorizontalFlip(),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        if not os.path.isdir(self.dataPath + setName):
            try:
                url = self.url + setName + ".zip"

                file = setName + ".zip"
                u = urllib2.urlopen(url)
                f = open(file, 'wb')
                file_size = int(u.info().getheaders("Content-Length")[0])

                file_size_dl = 0
                block_sz = 8192
                while True:
                    buff = u.read(block_sz)
                    if not buff:
                        break

                    file_size_dl += len(buff)
                    f.write(buff)
                    if ((file_size_dl * 100. / file_size)%10 == 0):
                        print(str(int(file_size_dl * 100. / file_size))+"%")

                f.close()
                print("Downloaded data set ("+setName+")")
                datazip = zipfile.ZipFile(setName+".zip")
                datazip.extractall(self.dataPath)
                datazip.close()
                os.remove(setName+".zip")
                print("Removed zip file")
            except:
                print("Error downloading data set " + setName, file=sys.stderr)
        self.dataPath += setName
        print(self.dataPath)
        self.x_files = sorted(glob.glob(os.path.join(self.dataPath, mode+"A") + '/*.*'))
        self.y_files = sorted(glob.glob(os.path.join(self.dataPath, mode+"B") + '/*.*'))
        self.loaded = 1
        print("Finished loading data")
        
    def __len__(self):
        if not self.loaded:
            print("Data set not loaded", file=sys.stderr)
        else:
            return max(len(self.x_files), len(self.y_files))
        
    def __getitem__(self, index):
        if not self.loaded:
            print("Data set not loaded", file=sys.stderr)
            return 0;
        else:
            x_img = self.transform(Image.open(self.x_files[index % len(self.x_files)]))
            y_img = self.transform(Image.open(self.y_files[random.randint(0, len(self.y_files) - 1)]))
            
            return {'x':x_img,'y':y_img}
    
