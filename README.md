# Cross-View Gait Based Human Idendification with Deep CNNs
A pytorch implementation of the Local-Bottom Network (LB) in the paper:


> Wu, Zifeng, et al. "A comprehensive study on cross-view gait based human identification with deep cnns." IEEE transactions on pattern analysis and machine intelligence 39.2 (2017): 209-226.



# Dependency
- ```python3```
- ```pytorch >= 0.4.0```
- [visdom](https://github.com/facebookresearch/visdom).
- [opencv](https://github.com/opencv/opencv)

# Model
- In ```src/model.py``` there are two models: LBNet and LBNet\_1. LBNet\_1 is more close to the model described in the section 4.2.1 of the original paper. You can select either one. The results are close to each other.

# Training

- To train the model, put the [CASIA-B dataset](http://kylezheng.org/gait-recognition/) silhoutte data under repository
- ```mkdir snapshot``` to build the directory for saving models
- goto the ```src``` dir and run
```
python3 train.py
```

The model will be saved into the execution dir every 10000 iterations. You can change the interval in train.py.

# Monitor the performance


- Install [visdom](https://github.com/facebookresearch/visdom).
- Start the visdom server with ```python3 -m visdom.server -port 5274``` or any port you like (change the port in train.py and test.py)
- Open this URL in your browser: `http://localhost:5274` You will see the training loss curve and the validation accuracy curve.

# Testing

- goto ```src``` dir and run ```python3 test.py```. You can select which snapshot to use by modifying the ```checkpoint = th.load('../snapshot/snapshot_75000.pth')``` to other snapshots. Be patient since it takes a long time. The computed similarities will be saved into ```similarity.npy```
- run ```python3 compute_acc_per_angle``` to compute the accuracy for each prove view and gallery view. The results will be saved into ```acc_table.csv```

You will get a table like this.

## LBNet

![table](https://github.com/xuehy/Cross-View-Gait-Deep-CNNs/blob/master/result.png)

## LBNet_1

![table1](https://github.com/xuehy/Cross-View-Gait-Deep-CNNs/blob/master/result1.png)
