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
        
    def downloadData(self, setName):
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
        
    def loadData(self, 
                 setName, 
                 mode, 
                 img_size,
                 im_transforms): 
        self.mode = mode;
        self.transform = transforms.Compose(im_transforms)
        self.dataPath += setName
        print(self.dataPath)
        self.x_files = sorted(glob.glob(os.path.join(self.dataPath, mode+"A") + '/*.*'))
        self.y_files = sorted(glob.glob(os.path.join(self.dataPath, mode+"B") + '/*.*'))
        self.loaded = 1
        print("Finished loading data")
        
    def loadImage(self, img_path, im_transforms, mode):
        self.mode = mode;
        self.transform = transforms.Compose(im_transforms);
        self.x_files = [img_path];
        self.loaded = 1
        print("Finished loading image")
        
    def loadImageSet(self, imgs_path, im_transforms, mode):
        self.mode = mode;
        self.transform = transforms.Compose(im_transforms);
        self.x_files = sorted(glob.glob(imgs_path + '/*.*'))
        self.loaded = 1
        print("Finished loading images")
        
    def __len__(self):
        if not self.loaded:
            print("Data set not loaded", file=sys.stderr)
        else:
            if (self.mode == "train"):
                return max(len(self.x_files), len(self.y_files))
            elif (self.mode == "test"):
                return len(self.x_files)
        
    def __getitem__(self, index):
        if not self.loaded:
            print("Data set not loaded", file=sys.stderr)
            return 0;
        else:
            x_img = self.transform(Image.open(self.x_files[index % len(self.x_files)]))
            if (self.mode == "train"):
                y_img = self.transform(Image.open(self.y_files[random.randint(0, len(self.y_files) - 1)]))

                return {'x':x_img,'y':y_img}
            elif (self.mode == "test"):
                return {'img':x_img,'path':os.path.basename(self.x_files[index % len(self.x_files)])}
