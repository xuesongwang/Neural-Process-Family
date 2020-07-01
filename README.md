
# Neural Process Families

This repository implements a pytorch version of Neural Process families:
- [Conditional Neural Processes](https://arxiv.org/abs/1807.01613) (CNP)

- [Neural Processes](https://arxiv.org/abs/1807.01622) (NP)

- [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) (ANP)

- [Convolutional Conditional Neural Processes](https://arxiv.org/abs/1910.13556) (ConvCNP)

The fitting process of each model is presented as below. An RBF-kernel and a periodic kerenl are adopted.
 
 
RBF kernel
<p align="center">
<img src="saved_fig/rbf-kernel-eq.png" width="50">
</p>
Periodic kernel
<p align="center">
<img src="saved_fig/periodic-kernel-eq.png" width="50">
</p>

CNP:
<p align="center">
<img src="saved_fig/CNP_EQ.gif" width="300"> <img src="saved_fig/CNP_period.gif" width="300">
</p>
NP:
<p align="center">
<img src="saved_fig/NP_EQ.gif" width="300"> <img src="saved_fig/NP_period.gif" width="300">
</p>
ANP:
<p align="center">
<img src="saved_fig/ANP_EQ.gif" width="300"> <img src="saved_fig/ANP_period.gif" width="300">
</p>
ConvCNP:
<p align="center">
<img src="saved_fig/ConvCNP_EQ.gif" width="300"> <img src="saved_fig/ConvCNP_period.gif" width="300">
</p>

## Requirements
* Python 3.6+
* Pytorch 1.4
* matplotlib 3.1.2
* tqdm 4.36.1+
* numpy 0.17.1
* pandas 0.25.1
* tensorboard 1.14+ (optional if you do not want to visualize the training process) 
    

To install the requirements!

```bash
pip install -r requirements.txt
```


## Training

#### Off-the-grid datasets
To train the model(s) for off-the-grid datasets, run this command:

```train
python train_1d.py --name EQ --epochs 200 --learning_rate 3e-4 --weight_decay 1e-5
```

The first argument, `name`(`default = EQ`), specifies the data that the model will be trained
on, and should be one of the following:
 
* `EQ`: samples from a GP with an exponentiated quadratic (EQ) kernel;
* `matern`: samples from a GP with a Matern-5/2 kernel;
* `period`: samples from a GP with a weakly-periodic kernel
* `smart_meter`: This dataset is referred from: https://github.com/3springs/attentive-neural-processes/tree/RANPfSD/data 
 To train on smart_meter, you need to change the argument `indir` in the function of `get_smartmeter_df` in `data/smart_meter.py` 
 to your own data path. 
     
#### On-the-grid datasets      
To train the models for on-the-grid datasets, run this command:
```train
python train_2d.py --dataset mnist --batch-size 16 --learning-rate 5e-4 --epochs 100
```           
The first argument, `dataset`(`default = mnist`), specifies the data that the model will be trained
on, and should be one of the following:
* `mnist`:This dataset can be downloaded using torchvision.datasets. Change the path `MNIST('./MNIST/mnist_data'...)`
of the function `load_dataset` in `data/image_data.py` to the location of your datasets;
* `svhn`: This dataset can also be downloaded using torchvision;
* `celebA`: This dataset can be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

We split the downloaded training datasets into training and validating sets with the proportion: 7:3 and use additional testing sets. 

## Evaluation
#### Off-the-grid datasets
To evaluate my model on off-the-grid datasets, run:

```eval
python eval_1d.py --name EQ
```
The argument is the same as in train_1d.py. A model called `name` + `_model.pt`
will be loaded from the folder `saved_model`

#### On-the-grid datasets
To evaluate my model on On-the-grid datasets, run:
```eval
python eval_2d.py --dataset mnist --batch-size 16
```
The argument is the same as in train_2d.py. A model called `dataset` + `_model.pth.gz`
will be loaded from the folder `saved_model`

## Results

Our model achieves the following log-likelihood (displayed in mean (variance)) on the on/off-the-grid datasets:



| Model name         | EQ              | Matern         |  Period       | Smart Meter   |
| ------------------ |---------------- | -------------- |-------------- | --------------|
| NP-PROV            | 2.20 (0.02)     |    0.90 (0.03) |  -1.00 (0.02) | 2.32 (0.05)   |

| Model name         | MNIST           | SVHN           |  celebA       | miniImageNet   |
| ------------------ |---------------- | -------------- |-------------- | -------------- |
| NP-PROV            | 2.66 (3e-2)     |    8.24 (5e-2) |  5.11 (1e-2)  | 4.39 (2e-1)    |

## References
* Official implementations (tensorflow) of (A)NP and CNP:
https://github.com/deepmind/neural-processes

* Official implementation of ConvCNP on 1d datasets:
https://github.com/cambridge-mlg/convcnp
 . Our reproduction of NP and CNP is inspired by this repo, except that we use the same GPsampler and evaluation as in 
 the official NP repo

* NP for sequential data, ANP-RNN： https://github.com/3springs/attentive-neural-processes

* Understanding Gaussian Process with visualizations: https://distill.pub/2019/visual-exploration-gaussian-processes/ 

Neural Process Papers I found useful:
