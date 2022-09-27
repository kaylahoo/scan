import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
t = torch.randn(2,2,3,3)*10//10
b,c,w,h =t.size()
print(t)
print(b,c,w,h)
t = t.view(2,2,9)
print(t)

tt = torch.reshape(t,(2,2,9))
print(tt)
print(tt.size())


m = torch.matmul(t,t.permute(0,2,1))
print(m)
print(m.size())


a = torch.from_numpy(np.array([[[1,2,3],[2,3,4]]]))
print(a.size())
sum = torch.sum(a,dim=(0,1,2),keepdim=False)
print(sum)

on = torch.ones((1,2*3))
print(on)
print(on.shape)


t = np.random.randint(1,10,[1,6,16]) # b,c,h*w
t = torch.from_numpy(t)
print(t)
sampling_pos = torch.multinomial(torch.ones(size=(1,4*4))*0.5,4)
print(sampling_pos)
sampling_pos = torch.unsqueeze(sampling_pos,dim=0).expand(1,6,4)

print(sampling_pos)

fast = torch.gather(t, dim=2, index=sampling_pos)
print(fast)
print(fast.size())



t = np.random.randint(1,10,[2,3,4,4]) # b,c,h*w
t = torch.from_numpy(t)
print(t)
t = t.reshape(2,3,1,4,4)
print(t)
print(t.size())
t = torch.tile(t,dims=[1,1,2,1,1])
print(t)
print(t.size())




samp = torch.multinomial(torch.ones(1,8)*0.5,4)
print(samp)
samp = torch.unsqueeze(samp,dim=2).expand(2,4,16)
print(samp)
tt = torch.gather(t,dim=1,index=samp)
print(tt)


#t = np.random.randint(1,10,[2,8,6,6]) # b,c,h*w
t = np.random.randn(2,4,3,3)
t = torch.from_numpy(t)
print(t)
print(t.size())
mean = torch.mean(t,dim=(-1,-2),keepdim=True)
mean = torch.squeeze(mean,dim=-1)
print(mean)
print(mean.size())


t = np.random.randint(1,10,[2,4,3]) # b,c,h*w
t = torch.from_numpy(t)
print(t)
print(t.size())
tt = torch.tile(t.view(2,4,3,1,1), dims=[1,1,1,2,2])
print(tt)
print(tt.size())


t = np.random.randint(1,10,[2,3,2,2]) # b,c,h*w
t = torch.from_numpy(t)
print(t)
print(t.size())
tt = np.random.randint(1,10,[2,3,1,1])
tt = torch.from_numpy(tt)
print(tt)
print(tt+t)
print((tt+t).size())


t = np.random.randint(1,10,[2,3,2,2]) # b,c,h*w
t = torch.from_numpy(t)
print(t)
tt = np.random.randint(1,10,[2,3,2,2]) # b,c,h*w
tt = torch.from_numpy(tt)
print(tt)
ttt = torch.cat([t,tt],dim=2)
print(ttt)
print(ttt.size())


sam = torch.tensor([1,1,0,0,1,0,1,0,0,1]).expand(2,1,10)
print(sam)
sampling_pos = torch.multinomial(sam[-1]*0.5, 4)
print(sampling_pos)
sampling_pos1 = sampling_pos.expand(2,3,4)
print(sampling_pos1)
print(sampling_pos1.size())

t = torch.from_numpy(np.random.randint(1,10,(2,1,4,4)))
print(t)
t = t.resize(4,4)
print(t)



from scipy.misc import imread
import scipy
mask = imread("D:/Study/Datasets/mask/irregular_mask/testing_mask_dataset/09093.png")
mask = scipy.misc.imresize(mask, [512, 512])
mask = (mask > 0).astype(int) * 2
print(mask.dtype)
np.savetxt('C:/Users/cityrain/Desktop/demo.txt',mask)
'''

t = torch.ones(1, 1, 2, 2)
print(t.size())
