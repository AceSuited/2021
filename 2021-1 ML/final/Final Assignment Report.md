# Final Assignment Report

**2016112083 김연웅**

## About My Model Design

![image-20210620173337967](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210620173337967.png)

### Model Layers

**time dataset - LSTM**

- Input size: 35
- hidden size: 10
- num_layers: 1
- output size: 32

**static dataset - FCN**

- input layer : 7 
- output layer : 32

**Activation function** : ReLU, Sigmoid(for binary classification)

**converged** **- FCN**

1st - input : 32 + 32 = 64,  output: 32

2nd (output)  - input: 32 , output: 1



I used fully-connected network for time-series features and LSTM network for static dataset. Then I converged into a fully network layer again, for classification task.

### Number of Parameters

![image-20210620131612417](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210620131612417.png)

## weighted F1(up to three decimal points)

![image-20210620171224390](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210620171224390.png)

**WEIGHTED F1: 0.942**

## Random Seeds Configuration

![image-20210521163147219](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521163147219.png)



