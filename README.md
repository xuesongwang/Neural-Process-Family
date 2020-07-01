
# Neural Process Families

This repository implements a pytorch version of Neural Process families including:
- [Conditional Neural Processes](https://arxiv.org/abs/1807.01613) (CNP)

- [Neural Processes](https://arxiv.org/abs/1807.01622) (NP)

- [Attentive Neural Processes](https://arxiv.org/abs/1901.05761) (ANP)

- [Convolutional Conditional Neural Processes](https://arxiv.org/abs/1910.13556) (ConvCNP)

<p align="center">
<img src="demo_images/NP-PROV-MU.jpg" width="200"> <img src="demo_images/NP-PROV-SIGMA.jpg" width="200">
</p>

## Requirements
* Python 3.6 or higher.

* `gcc` and `gfortran`:
    On OS X, these are both installed with `brew install gcc`.
    On Linux, `gcc` is most likely already available,
    and `gfortran` can be installed with `apt-get install gfortran`.
    

Install the requirements and You should now be ready to go!

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

Neural Process Papers I found useful:
