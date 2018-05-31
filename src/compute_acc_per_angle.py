import numpy as np
from operator import itemgetter
import csv

sim = np.load('./similarity.npy')
acc_table = np.zeros((11, 11))
k_for_knn = 3
for p_angle in range(11):
    for g_angle in range(11):
        correct = 0
        total = 0
        for pid in range(50):
            for pset in range(2):
                # pid x gallery_size
                vote = np.zeros((50, ))
                sim_ = sim[pid, :, pset, :, p_angle, g_angle]
                zipped = []
                for i in range(50):
                    a = zip([i for j in range(4)], list(sim_[i]))
                    zipped += a
                sim_list = sorted(zipped, reverse=True, key=itemgetter(1))
                sim_list = sim_list[0:k_for_knn]
                for item in sim_list:
                    vote[item[0]] += 1
                predict = np.argmax(vote)
                if predict == pid:
                    correct += 1
                total += 1
        acc_table[g_angle][p_angle] = correct / total
        print('probe angle = {}, gallery angle = {}, acc = {}'.format(
            p_angle * 18, g_angle * 18, correct/total
        ))

with open('acc_table.csv', 'wt') as output:
    output = csv.writer(output)
    for i in range(11):
        line = list(acc_table[i])
        output.writerow(line)
