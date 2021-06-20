# HW 4 Report

**2016112083 김연웅**



# About My Model Design

![image-20210610005232153](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210610005232153.png)

### Model Layers

1. LSTM

- Input size: 5
- hidden size: 5
- num_layers: 1
- output size: 5

2. **Activation function** : ReLU

3. Fully Connected Layer

   1st : 128

   2nd(output) : 128

### Number of Parameters

![image-20210610005609824](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210610005609824.png)

**The number of parameters: 1653 elements.**

## accuracy in terms of the mean absolute percentage error (MAPE)

![image-20210610005657172](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210610005657172.png)

### ![image-20210610005721985](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210610005721985.png)

### Hyper parameter setting

num epochs : 100

Learnint rate :  0.001

batch size : 32

### Random Seeds Configuration

![image-20210521163147219](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521163147219.png)

