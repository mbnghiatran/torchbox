# Torchbox - a fully customizable deep learning implementation solution in Pytorch
implement and fine-tune your deep learning model in an easy, customizable and fastest way

First version created in Oct 2019 by @lexuanhieu

# Introduction

Several problems that deep learning researchers usually face are that: 

1. Everytime they train a model, they have to replicate or copy&paste many 
resuable code, which is time-expensive

2. The code is unstructed and unclean, making it more difficult to read and interpret

3. The work is messy and hardly replicable. 

4. To few metrics

I therefore release torchbox, a Deep Learning implementation tool for students, and scientists that can tackle these aforementioned problems.

Torchbox provides you a fast, clean, structured and repeatable implementation and fine-tuning of your custom Deep Learning models. Moreoever, you can add as many metrics as you can to measure your model, all you need to provide is the metric's class and its methods. For example, we already integrated all of Sklearn metrics, all you need to do is provide the metric method names, example: f1_score, precision_score, etc... No more painful metrics coding!!!

# What it provides

### Modularity and customizability

Every components you need in training is a separated module. 

### Scalable project

### All in one config file

Introduce the usage of a config file that contains neccessary params for training. Everything you need is just modify this file in order to train

### Simple training and testing 

Only ``` $ python train.py```  is required to do so

### Specilized for transfer learning

I have written a class wrapper that can make it easier for you to turn on transfer learning. Instead of importing a pretrained model, excluding
its head, all you need is just writting down it names and customize your new classifer (head).

List of models are available here :

### Attach your custom metrics easily

Suppose you have implemented a custom metrics, for example the Thresholded Jaccard Similarity score in the ISIC 2019 Segmentation competition, you can use it to measure your model without the pain of editting/typing too much code. Simply inputing the metric class name and its method to use

# Quick start

## Example Usage

I have already implemented a built-in classification example. To use it:

1. Starting by cloning this github into your machine

2. CD to the project folder and install dependencies 

3. Editting config file as needed (See documentation)

4. Run : ```$ python train.py```


# Documentation

## The config files
Location : cfg/tenes.json

The config file is the core file in this package. It contains core parameters to train. While most of them are interpretable, I provide here docs for some important params.

### **sess_name**
```json
"session": {
  "project_name": "trial_1",
  "save_path": "logs/"
},
```
This will create a directory that have name is ["save_path" / "project_name" + time]

### **data params**
```json
"data":{
  "data.class": "data_loader.Classify.ClassificationDataset",
  "data_csv_name": "dataset/train.csv",
  "validation_ratio": "0.2",
  "test_csv_name": "dataset/test.csv",
  "label_dict": ["cat","dog"],
  "batch_size": 4,
}
```
In order to train, we requires two csv files for the training and testing set, each has **two columns named "file_name" and "label (int starting from 0)"**

Kindly provide the two csv file path in "data_csv_name" (for training set) and "test_csv_name" (for testing set)

The training set will be further splitted to a smaller validation set by the **validation_ratio**

*label_dict:* a list of all label names for mapping

### **Optimizer**
```json
"optimizer": {
  "name": "Adam",
  "lr": 1e-4,
}
```
*name:* name of the optimizer class in **torch.optim**. *Ex*: If you intend to use **torch.optim.Adam** simply type "Adam". 

*lr:* Args for optimizer

### **Learning rate Scheduler**
```json
"scheduler": {
  "name": "ReduceLROnPlateau",
  "min_lr": 1e-5,
  "mode": "min",
  "patience": 3,
  "factor": 0.1,
}
```
*name* : name of the optimizer class in **torch.optim.lr_scheduler**. *Ex*: if you intend to use **torch.optim.lr_scheduler.ReduceLROnPlateau** simply type "ReduceLROnPlateau". 

other args: Args for decay learning rate 

## **Model**
```json
"model":{
  "model.class": "models.resnet_transfer.resnet.ResNet_transfer",
  "model_name": "resnet50",
  "num_class": 2,
  "pretrained": true,
}
```
*model.class:* module ResNet_transfer is defined in models/resnet_transfer/resnet.py

*model_name:* name of feature extractor

*num_class:* number of class

**Note:** If you want to customizing your own model, you should located them in the **models** folder and define *model.class*


### **Other setting to train**
```json
"train": {
  "num_epoch": 20,
  "loss": "CrossEntropyLoss",
  "metrics": ["accuracy_score"],
  "early_patience": 5,
  "mode": "min",
}
```
*num_epoch:* number of epoch.

*loss:* if you intend to use torch.nn.CrossEntropyLoss simply type CrossEntropyLoss.

*early_patience*, *mode:* Args for Early Stopping


## **Customizing your metrics: use your own metrics**

The **utils/metrics.py** file is a wrap file that handles the metrics implementation.

Let's say you want to use your custom and newly created metrics. All you need to do is implement a Class which has a method to take in (labels [numpy or list type],preds) and calculate your measurement.

Example: MyMetrics.IoU(labels,preds) 

Then in the utils/metrics_loader.py file:

1. import your class

```python
    do_metric = getattr(
         skmetrics, metric, "The metric {} is not implemented".format(metric)
    )
```

2. Replace the skmetrics with your class, for ex: MyMetrics

3. Editing metrics name in your config file, example: `"metrics" : ["IoU"]`


## **Customizing tranformation(augmentation) and dataloaders**

See data_loader/dataloader.py for customizing dataloaders

See data_loader/transform.py for customizing the transformation

## **Customizing training actions:**

See in train.py,
```
  if epoch == 5:
    for param in model.parameters():
        param.requires_grad = True
```
In the first few epochs, model is freezed the **feature_extractor** to train **classify layer** and then they will be unfreezed train entire model.

