import torch as th
import cv2
import numpy as np
import os
import itertools
from torch.utils.data import Dataset


def loadImage(path):
    inImage = cv2.imread(path, 0)
    info = np.iinfo(inImage.dtype)
    inImage = inImage.astype(np.float) / info.max
    iw = inImage.shape[1]
    ih = inImage.shape[0]
    if iw < ih:
        inImage = cv2.resize(inImage, (126, int(126 * ih/iw)))
    else:
        inImage = cv2.resize(inImage, (int(126 * iw / ih), 126))
    inImage = inImage[0:126, 0:126]
    return th.from_numpy(2 * inImage - 1).unsqueeze(0)


class DatasetForTrain():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ids = np.arange(1, 51)
        self.cond = ['bg-01', 'bg-02', 'cl-01', 'cl-02',
                     'nm-01', 'nm-02', 'nm-03', 'nm-04',
                     'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072',
                       '090',
                       '108', '126', '144', '162', '180']
        self.n_cond = len(self.cond)
        self.n_ang = len(self.angles)
        self.n_id = 50

    def getbatch(self, batchsize):
        batch1 = []
        batch2 = []
        batch3 = []  # store labels
        for i in range(batchsize):
            seed = th.randint(1, 100000, (1,)).item()
            th.manual_seed((i+1)*seed)

            path1 = None
            while True:
                id1 = th.randint(0, self.n_id, (1,)).item() + 1
                id1 = '%03d' % id1
                cond1 = th.randint(0, self.n_cond, (1,)).item()
                cond1 = int(cond1)
                cond1 = self.cond[cond1]
                angle1 = th.randint(0, self.n_ang, (1,)).item()
                angle1 = int(angle1)
                angle1 = self.angles[angle1]

                path1 = self.data_dir + id1 + '/' + cond1 + '/' + id1 + '-' + \
                    cond1 + '-' + angle1 + '.png'
                if os.path.exists(path1):
                    break
            while True:
                if i % 2 == 1:  # positive
                    id2 = id1
                else:  # negative
                    id2 = th.randint(0, self.n_id, (1,)).item() + 1
                    id2 = '%03d' % id2
                    while id2 == id1:
                        id2 = th.randint(0, self.n_id, (1,)).item() + 1
                        id2 = '%03d' % id2
                cond2 = th.randint(0, self.n_cond, (1,)).item()
                cond2 = int(cond2)
                cond2 = self.cond[cond2]
                angle2 = th.randint(0, self.n_ang, (1,)).item()
                angle2 = int(angle2)
                angle2 = self.angles[angle2]
                path2 = self.data_dir + id2 + '/' + cond2 + '/' + id2 + '-' + \
                    cond2 + '-' + angle2 + '.png'
                if os.path.exists(path2):
                    break
            img1 = loadImage(path1)
            img2 = loadImage(path2)
            batch1.append(img1)
            batch2.append(img2)
            batch3.append([i % 2])
        return th.stack(batch1), th.stack(batch2), th.Tensor(batch3)


class DatasetForEval(Dataset):
    def __init__(self, data_dir):
        super(DatasetForEval, self).__init__()
        self.data_dir = data_dir
        self.ids = np.arange(51, 75)
        self.gallery_cond = ['nm-01', 'nm-02', 'nm-03', 'nm-04']
        self.probe_cond = ['nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072',
                       '090',
                       '108', '126', '144', '162', '180']
        self.n_ang = len(self.angles)

        self.all_possible_paths_g = []
        self.all_possible_paths_p = []
        for idx in self.ids:
            for cond in self.gallery_cond:
                for angle in self.angles:
                    index = '%03d' % idx
                    path = index + '/' + cond + '/' + index + '-' + \
                        cond + '-' + angle + '.png'
                    if os.path.exists(self.data_dir + path):
                        self.all_possible_paths_g.append(path)

            for cond in self.probe_cond:
                for angle in self.angles:
                    index = '%03d' % idx
                    path = index + '/' + cond + '/' + index + '-' + \
                        cond + '-' + angle + '.png'
                    if os.path.exists(self.data_dir + path):
                        self.all_possible_paths_p.append(path)

        self.all_possible_pairs = list(
            itertools.product(self.all_possible_paths_p,
                              self.all_possible_paths_g))
        self.ptr = 0
        print('Evaluation set prepared')

    def __len__(self):
        return len(self.all_possible_pairs)

    def __getitem__(self, idx):
        pair = self.all_possible_pairs[idx]
        img1 = loadImage(self.data_dir + pair[0])
        img2 = loadImage(self.data_dir + pair[1])
        label = 1 if pair[0][0:3] == pair[1][0:3] else 0
        PROBE_ANGLE = pair[0][-7:-4]
        return img1, img2, label, [int(pair[0][0:3]), int(pair[1][0:3])], \
            PROBE_ANGLE


class DatasetForTrainWithLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ids = np.arange(1, 51)
        self.cond = ['bg-01', 'bg-02', 'cl-01', 'cl-02',
                     'nm-01', 'nm-02', 'nm-03', 'nm-04',
                     'nm-05', 'nm-06']
        self.angles = ['000', '018', '036', '054', '072',
                       '090',
                       '108', '126', '144', '162', '180']
        self.n_cond = len(self.cond)
        self.n_ang = len(self.angles)
        self.n_id = 50
        self.seed = th.randint(1, 100000, (1,)).item()
        th.manual_seed((100)*self.seed)

    def __len__(self):
        return 2000000 * 128

    def __getitem__(self, idx):
        path1 = None
        while True:
            id1 = th.randint(0, self.n_id, (1,)).item() + 1
            id1 = '%03d' % id1
            cond1 = th.randint(0, self.n_cond, (1,)).item()
            cond1 = int(cond1)
            cond1 = self.cond[cond1]
            angle1 = th.randint(0, self.n_ang, (1,)).item()
            angle1 = int(angle1)
            angle1 = self.angles[angle1]

            path1 = self.data_dir + id1 + '/' + cond1 + '/' + id1 + '-' + \
                cond1 + '-' + angle1 + '.png'
            if os.path.exists(path1):
                break
        while True:
            if idx % 2 == 1:  # positive
                id2 = id1
            else:  # negative
                id2 = th.randint(0, self.n_id, (1,)).item() + 1
                id2 = '%03d' % id2
                while id2 == id1:
                    id2 = th.randint(0, self.n_id, (1,)).item() + 1
                    id2 = '%03d' % id2
            cond2 = th.randint(0, self.n_cond, (1,)).item()
            cond2 = int(cond2)
            cond2 = self.cond[cond2]
            angle2 = th.randint(0, self.n_ang, (1,)).item()
            angle2 = int(angle2)
            angle2 = self.angles[angle2]
            path2 = self.data_dir + id2 + '/' + cond2 + '/' + id2 + '-' + \
                cond2 + '-' + angle2 + '.png'
            if os.path.exists(path2):
                break
        img1 = loadImage(path1)
        img2 = loadImage(path2)
        return img1, img2, int(idx % 2)
