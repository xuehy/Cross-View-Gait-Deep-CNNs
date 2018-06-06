import torch as th
from model import LBNet_1
import time
from dataset import DatasetForTest, loadImage
import numpy as np
import sys


val_acc = None
lbnet = LBNet_1()
device = th.device("cuda:0")
checkpoint = th.load('../snapshot/snapshot_80000.pth')
lbnet = lbnet.to(device)
lbnet.load_state_dict(checkpoint['model'])
lbnet.eval()

bs = 128
testset = DatasetForTest('../data/GEI_CASIA_B/gei/')
print('test probe count = {}'.format(len(testset.all_possible_paths_p)))
print('Start Testing...')
localtime = time.asctime(time.localtime(time.time()))
print('Evaluation starts at {}'.format(localtime))
sim = np.zeros((50, 50, 2, 4, 11, 11))  # probe person,
# gallery person, probe set, gallery set, probe angle, gallery_angle
count = 0
for path in testset.all_possible_paths_p:
    sys.stdout.write(
        "processing probe %s [%d/%d] \r" % (path,
                                            count+1,
                                            len(
                                                testset.all_possible_paths_p
                                            )))
    sys.stdout.flush()
    pdir = testset.data_dir + path
    img1 = loadImage(pdir).unsqueeze(0)
    for path_g in testset.all_possible_paths_g:
        gdir = testset.data_dir + path_g
        img2 = loadImage(gdir).unsqueeze(0)
        img = th.cat((img1, img2), 1)
        img = img.to(device).to(th.float32)
        output = lbnet(img)
        id1 = int(path.split('/')[0]) - 75
        p_angle = int(path.split('/')[2][-7:-4]) // 18
        p_set = int(path.split('/')[1][-2::]) - 5
        id2 = int(path_g.split('/')[0]) - 75
        g_angle = int(path_g[-7:-4]) // 18
        g_set = int(path_g.split('/')[1][-2::]) - 1
        sim[id1][id2][p_set][g_set][p_angle][g_angle] = output.cpu().item()

    count += 1
localtime = time.asctime(time.localtime(time.time()))
print('Evaluation ends at {}'.format(localtime))

np.save('similarity.npy', sim)
print('Saved similarities into similarity.npy')
