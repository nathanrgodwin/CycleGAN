
# coding: utf-8

# <h1>CycleGAN for Style Transfer</h1>
# <h3> Imports </h3>

# In[ ]:


#Python imports
import torch
import torch.nn as tnn
from torch.autograd import Variable
import sys
import itertools
from torch.utils.data import DataLoader
from Dataset import ImageSet
import datetime
import torchvision.utils as tv
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image

#Import network elements
import network_elements as net

#stdout code blocks reload's print statement
#prevstd = sys.stdout
#sys.stdout = None
reload(net)
print("Reloading net")
#sys.stdout = prevstd


# <h3>Set parameters</h3>

# In[ ]:


#The number of channels in the input image
im_in_channels = 3

#The number of channels in the output image
im_out_channels = 3

#The size of the largest size of the input image
im_size = 128

LEARNING_RATE = 0.0002

LR_DECAY_START = 100
LR_DECAY_END = 200
NUM_EPOCHS = 200

load_partial_net = True
CURR_EPOCH = 0


# <h3>Import dataset</h3>

# In[ ]:


# apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
imageSet = ImageSet();
dataset = "ukiyoe2photo"
imageSet.downloadData(dataset)
training_transforms = [transforms.Resize(int(im_size*1.12), Image.BICUBIC),
                  transforms.RandomCrop(im_size),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))];
imageSet.loadData(dataset, 'train', im_size, training_transforms)
imgLoader = DataLoader(imageSet,shuffle=True)


# <h3>Define the CycleGAN network</h3>

# In[ ]:


#G : X->Y
G = net.Mapping(im_in_channels, im_out_channels, im_size)

#F : Y->X
F = net.Mapping(im_out_channels, im_out_channels, im_size)

#G mapping discriminator
Dx = net.Discriminator(im_in_channels)

#F mapping discriminator
Dy = net.Discriminator(im_out_channels)

#Initialize CUDA implementations
G.cuda();
F.cuda();
Dx.cuda();
Dy.cuda();

#Define losses
cycle_loss = tnn.L1Loss(); #Cycle-consistency loss
gan_loss = tnn.MSELoss(); #Adversarial loss
#identity_loss = tnn.L1Loss(); #Prevents image tinting


# <h3>Initialize weights for each network</h3>

# In[ ]:


# norm_weights function applies Gaussian norm weights with mean 0 and stddev 0.02
G.apply(net.conv_norm_weights);
F.apply(net.conv_norm_weights);
Dx.apply(net.conv_norm_weights);
Dy.apply(net.conv_norm_weights);


# <h3>Set optimizer parameters</h3>

# In[ ]:


gan_opt = torch.optim.Adam(itertools.chain(G.parameters(), F.parameters()),
                           lr = LEARNING_RATE)
dx_opt = torch.optim.Adam(Dx.parameters(), lr = LEARNING_RATE)
dy_opt = torch.optim.Adam(Dy.parameters(), lr = LEARNING_RATE)


# The paper uses Adam optimizer with a learning rate of 0.0002 for the first 100 epochs that linearly decays to 0 over the next 100 epochs.

# <h3> Initialize training variables </h3>

# In[ ]:


x = torch.cuda.FloatTensor(1, im_in_channels, im_size, im_size)
y = torch.cuda.FloatTensor(1, im_out_channels, im_size, im_size)

# Variable wrapper is depricated in >0.4
dReal = Variable(torch.cuda.FloatTensor(1).fill_(1.0), requires_grad = False)
dFake = Variable(torch.cuda.FloatTensor(1).fill_(0.0), requires_grad = False)

buffX_Y = net.MapBuffer()
buffY_X = net.MapBuffer()

if not os.path.isdir("./img/"):
    os.mkdir("./img/")

if not os.path.isdir("./img/" + dataset + "_noident/"):
    os.mkdir("./img/" + dataset + '_noident/')
    
if not os.path.isdir("./log/"):
    os.mkdir("./log")
    
if not os.path.isdir("./log/" + dataset + "_noident/"):
    os.mkdir("./log/" + dataset + "_noident/")
    
if os.path.isfile("./log/" + dataset + "_noident/loss.log") and not load_partial_net:    
    os.remove("./log/" + dataset + "_noident/loss.log")
    
logfile = open("./log/" + dataset + "_noident/loss.log","a")


# <h3> Load Partial Net </h3>

# In[ ]:


if (load_partial_net):
    if os.path.isfile("./model/" + dataset + "_noident/model_info.txt"):
        model_file = open("./model/" + dataset + "_noident/model_info.txt", "r")
        CURR_EPOCH = int(model_file.read())
        model_file.close()
    G.load_state_dict(torch.load('./model/' + dataset + '_noident/G.data'))
    F.load_state_dict(torch.load('./model/' + dataset + '_noident/F.data'))
    Dx.load_state_dict(torch.load('./model/' + dataset + '_noident/Dx.data'))
    Dy.load_state_dict(torch.load('./model/' + dataset + '_noident/Dy.data'))
    gan_opt.load_state_dict(torch.load('./model/' + dataset + '_noident/gan_opt.data'))
    dx_opt.load_state_dict(torch.load('./model/' + dataset + '_noident/dx_opt.data'))
    dy_opt.load_state_dict(torch.load('./model/' + dataset + '_noident/dy_opt.data'))
    
last_epoch = -1
if (load_partial_net):
    last_epoch = CURR_EPOCH-1

gan_lr = torch.optim.lr_scheduler.LambdaLR(gan_opt, net.LRDecay(LR_DECAY_START, LR_DECAY_END).step, last_epoch)
dx_lr = torch.optim.lr_scheduler.LambdaLR(dx_opt, net.LRDecay(LR_DECAY_START, LR_DECAY_END).step, last_epoch)
dy_lr = torch.optim.lr_scheduler.LambdaLR(dy_opt, net.LRDecay(LR_DECAY_START, LR_DECAY_END).step, last_epoch)


# <h3> Training </h3>

# In[ ]:


start_time = datetime.datetime.now()
j_prev = 0;
for i in range(CURR_EPOCH,NUM_EPOCHS):
    elapsed_time = datetime.datetime.now()-start_time
    print("Epoch %d/%d, %s"%(i,LR_DECAY_END,str(elapsed_time)))    
    for j, img in enumerate(imgLoader):
	if int((10.0*j)/len(imageSet)) != j_prev:
		j_prev = int((10.0*j)/len(imageSet));
		print(str(j_prev*10) + "%")
        src_x = Variable(x.copy_(img['x']))
        src_y = Variable(y.copy_(img['y']))
        
        gan_opt.zero_grad()
        
        #identity Loss
#        mapY_Y = G(src_y)
#        lossY_Y = identity_loss(mapY_Y, src_y)*5.0
        
#        mapX_X = F(src_x)
#        lossX_X = identity_loss(mapX_X, src_x)*5.0
        
        #GAN Loss
        mapX_Y = G(src_x) #Generate image Y' from image X
        dX_Y = Dy(mapX_Y) #Check if image Y' seems real
        lossX_Y = gan_loss(dX_Y, dReal) #Calculate Y' realism error
        
        mapY_X = F(src_y) #Generate image X' from image Y
        dY_X = Dx(mapY_X) #Check if image X' seems real
        lossY_X = gan_loss(dY_X, dReal) #Calculate X' realism error
        
        #Cycle Consistency Loss
        mapX_Y_X = F(mapX_Y)
        lossX_Y_X = cycle_loss(mapX_Y_X, src_x)*10.0
        
        mapY_X_Y = G(mapY_X)
        lossY_X_Y = cycle_loss(mapY_X_Y, src_y)*10.0
        
        mapLoss = lossX_Y + lossY_X + lossX_Y_X + lossY_X_Y
        mapLoss.backward()
        
        gan_opt.step()
        
        #Dx
        dx_opt.zero_grad()
        
        dX_res = Dx(src_x)
        lossX = gan_loss(dX_res, dReal)
        
        mapY_X_old = buffY_X.cycle(mapY_X)
        dY_X = Dx(mapY_X_old.detach())
        lossY_X = gan_loss(dY_X, dFake)
        
        dxLoss = (lossX + lossY_X)*0.5
        dxLoss.backward()
        
        dx_opt.step()
        
        #Dy
        dy_opt.zero_grad()
        
        dY_res = Dy(src_y)
        lossY = gan_loss(dY_res, dReal)
        
        mapX_Y_old = buffX_Y.cycle(mapX_Y)
        dX_Y = Dy(mapX_Y_old.detach())
        lossX_Y = gan_loss(dX_Y, dFake)
        
        dyLoss = (lossY + lossX_Y)*0.5
        dyLoss.backward()
        
        dy_opt.step()     
        
        #print("%d,%d,%f,%f,%f,%f,%f\n"%(i,j,mapLoss.data[0], 
        #                                  (lossX_X+lossY_Y).data[0], 
        #                                  (lossX_Y+lossY_X).data[0], 
        #                                  (lossX_Y_X+lossY_X_Y).data[0], 
        #                                  (dxLoss+dyLoss).data[0]))
        
        logfile.write("%d,%d,%f,%f,%f,%f\n"%(i,j,mapLoss.data[0], 
         #                                 (lossX_X+lossY_Y).data[0], 
                                          (lossX_Y+lossY_X).data[0], 
                                          (lossX_Y_X+lossY_X_Y).data[0], 
                                          (dxLoss+dyLoss).data[0]))
        
    tv.save_image(src_x.data, './img/' + dataset + '_noident/src_x_%d.jpg'%(i))
    tv.save_image(src_y.data, './img/' + dataset + '_noident/src_y_%d.jpg'%(i))
    tv.save_image(mapX_Y.data, './img/' + dataset + '_noident/mapX_Y_%d.jpg'%(i))
    tv.save_image(mapY_X.data, './img/' + dataset + '_noident/mapY_X_%d.jpg'%(i))
    gan_lr.step()
    dx_lr.step()
    dy_lr.step()
    

    if not os.path.isdir("./model/"):
        os.mkdir("./model/")
    
    if not os.path.isdir("./model/" + dataset + "_noident/"):
        os.mkdir("./model/" + dataset + "_noident/")
    model_file = open("./model/" + dataset + "_noident/model_info.txt", "w")
    model_file.write("%d"%(i+1))
    model_file.close()
    torch.save(G.state_dict(), './model/' + dataset + '_noident/G.data')
    torch.save(F.state_dict(), './model/' + dataset + '_noident/F.data')
    torch.save(Dx.state_dict(), './model/' + dataset + '_noident/Dx.data')
    torch.save(Dy.state_dict(), './model/' + dataset + '_noident/Dy.data')
    torch.save(gan_opt.state_dict(), './model/' + dataset + '_noident/gan_opt.data')
    torch.save(dx_opt.state_dict(), './model/' + dataset + '_noident/dx_opt.data')
    torch.save(dy_opt.state_dict(), './model/' + dataset + '_noident/dy_opt.data')


# In[ ]:


logfile.close()


# In[ ]:


print("Finished training")

