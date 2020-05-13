# Autodetection

This repository is the implementation for Autodetection by Jiuhong Xiao in New York University

**Autodetection: An End-to-end Autonomous Driving Detection System**

Jiuhong Xiao, Xinmeng Li, Junrong Zha

DEEP LEARNING 20SPRING (CSCI-GA 2572) final project

![img](framework.jpg)

## Usage

### Dependencies

This work depends on the following library:

Python == 3.8

Pytorch == 1.4.0

### Train and Validate

Data should be located at dataset/data/

Run all code in the root directory, e.g., to run trainBothFPN.py

```
python training/trainBothFPN.py
```

If you want to train LSTMmodel, make sure step_sizes in helper.py and bothModelLSTM.py are the same and scene_batch_size in trainBothLSTM.py and bothModelLSTM.py (batch_size in trainModel function) are the same.

## Results

<img src="results/result (1).jpg" alt="img" style="zoom: 50%;" /><img src="results/result (2).jpg" alt="img" style="zoom:50%;" />



<img src="results/result (3).jpg" alt="img" style="zoom:50%;" /><img src="results/result (4).jpg" alt="img" style="zoom:50%;" />

## Acknowledgemet

The implemention of EfficientNet and YOLOv3 layeris based on Lukemelas's implemention https://github.com/lukemelas/EfficientNet-PyTorch.git and Eriklindernoren's implementation https://github.com/eriklindernoren/PyTorch-YOLOv3.git with some revisement.
Thank you very much!
