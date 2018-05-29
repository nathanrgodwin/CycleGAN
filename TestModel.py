from __future__ import print_function
import sys
from PIL import Image
import os.path
from Parameters import Params
from torchvision import transforms
from torchvision import utils as tv
from Dataset import ImageSet
import torch
from torch.utils.data import DataLoader
from network_elements import Mapping
from torch.autograd import Variable

args = sys.argv;
path = args[-1];
model = args[-2];

p = Params();
p.model = model;
if "-r" in args[1:-2]:
    p.dir = 1
if "-o" in args[1:-2]:
    outputInd = args.index("-o");
    p.outputPath = args[outputInd+1];
else:
    if not os.path.exists("./outputs/"):
        os.mkdir("./outputs/")
    if p.dir:
        tempPath = path.split();
        p.outputPath = "./outputs/"+tempPath[-1]+"/";
        if not os.path.exists(p.outputPath):
            os.mkdir(p.outputPath);
    elif not p.dir:
        tempPath = os.path.dirname(path)+"/out_"+os.path.basename(path);
        p.outputPath = tempPath;
if "-d" in args[1:-2]:
    outputInd = args.index("-d");
    p.direction = args[outputInd+1];
        
print("Model under test is " + p.model);
print("The output path is "+p.outputPath);

images = ImageSet();
test_transforms = [transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ];
if (p.dir):
    if os.path.exists(path):
        print("Processing set in directory: "+path);
        set
    else:
        print("No directory exists at path: "+path, file=sys.stderr); 
    images.loadImageSet(path, test_transforms, "test");
    
else:
    if os.path.exists(path):
        print("Processing file at path: " + path);
    else:
        print("No file exists at path: "+path, file=sys.stderr);
    images.loadImage(path, test_transforms, "test");

imgLoader = DataLoader(images, 1, shuffle=False);

F = Mapping(3,3,128);
F.cuda();
F.load_state_dict(torch.load("./model/"+p.model+"/F.data"));
F.eval();
    
for i, img in enumerate(imgLoader):
    imgsize = img['img'].size();
    print(imgsize[2], imgsize[3]);
    img_gpu = torch.cuda.FloatTensor(1,3,imgsize[2],imgsize[3]);
    img_var = Variable(img_gpu.copy_(img['img']))
    result = 0.5*(F(img_var).data+1.0);
    tv.save_image(result, p.outputPath);