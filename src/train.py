import torch as th
import torch.nn as nn
from model import LBNet_1
import visdom
import time
import torch.optim as optim
import torch.nn.init as init
from dataset import DatasetForEval, DatasetForTrainWithLoader, loadImage
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


vis = visdom.Visdom(port=5274)
train_loss = None
val_acc = None
lbnet = LBNet_1()
device = th.device("cuda:1")

for mod in list(lbnet.children())[0].children():
    if isinstance(mod, nn.Conv2d):
        init.normal_(mod.weight, 0.0, 0.01)
        init.constant_(mod.bias, 0.0)

lbnet = lbnet.to(device)
lbnet.train()

iteration = 0
bs = 128
trainset = DatasetForTrainWithLoader(
    '../data/GEI_CASIA_B/gei/')
evalset = DatasetForEval('../data/GEI_CASIA_B/gei/')
print('evaluation probe count = {}'.format(len(evalset.all_possible_paths_p)))
print('evaluation size = {}'.format(evalset.__len__()))
trainset = DataLoader(trainset, bs, num_workers=8)
optimizer = optim.Adam(lbnet.parameters(), lr=0.0001)

print('Start Training...')
for iteration, data in enumerate(trainset):
    img1, img2, label = data
    lbnet.train()
    label = label.to(device).to(th.float32).unsqueeze(1)
    img1 = img1.to(device).to(th.float32)
    img2 = img2.to(device).to(th.float32)
    img = th.cat((img1, img2), 1)
    output = lbnet(img)
    loss = F.binary_cross_entropy(output, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iteration % 5 == 0:
        if train_loss is None:
            train_loss = vis.line(X=np.array([[iteration]]),
                                  Y=np.array([[loss.cpu().item()]]),
                                  opts=dict(
                                      title='LBNet training curve',
                                      ylabel='loss',
                                      xlabel='iterations'
                                  ))
        else:
            vis.line(X=np.array([[iteration]]),
                     Y=np.array([[loss.cpu().item()]]),
                     win=train_loss,
                     update='append')

    if (iteration + 1) % 5000 == 0:
        localtime = time.asctime(time.localtime(time.time()))
        print('Evaluation starts at {}'.format(localtime))
        acc = 0
        n = 0
        lbnet.eval()
        max_iter = evalset.__len__()

        for path in evalset.all_possible_paths_p:
            knn = np.zeros((24,))
            pdir = evalset.data_dir + path
            img1 = loadImage(pdir).unsqueeze(0)
            for path_g in evalset.all_possible_paths_g:
                gdir = evalset.data_dir + path_g
                img2 = loadImage(gdir).unsqueeze(0)
                img = th.cat((img1, img2), 1)
                img = img.to(device).to(th.float32)
                output = lbnet(img)
                output = output > 0.5

                id1 = int(path.split('/')[0]) - 51
                id2 = int(path_g.split('/')[0]) - 51
                if output == 1:
                    knn[id2] += 1

            predict = knn.argmax()
            if predict == id1:
                acc += 1
            n += 1

        if val_acc is None:
            val_acc = vis.line(X=np.array([[iteration]]),
                               Y=np.array([[acc/n]]),
                               opts=dict(
                                   title='Accuracy on Validation Set',
                                   xlabel='iteration',
                                   ylabel='accuracy'
                               ))
        else:
            vis.line(X=np.array([[iteration]]),
                     Y=np.array([[acc/n]]),
                     win=val_acc,
                     update='append')
        localtime = time.asctime(time.localtime(time.time()))
        print('{}, iter = {}, loss = {}, sim acc = {}'.format(
            localtime, iteration, loss.cpu().item(), acc/n))
    if (iteration + 1) % 10000 == 0:
        state = {
            'model': lbnet.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        th.save(state, '../snapshot/snapshot_{}.pth'.format(iteration+1))
    if iteration > 2000000:
        break
