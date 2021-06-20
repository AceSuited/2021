# HW 4 Report

**2016112083 김연웅**



# About My Model Design

![image-20210521161439957](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521161439957.png)

### Model Layers

1. **Input layer:** 

- Input dimension: 100

- output dimension: 30

2. **Frist hidden layer:**

- Input dimension : 30

- output dimension : 15
- activation function: ReLU
- dropout: 0.2

3. **Second hidden layer:**

- input dimension: 15
- output dimension: 1
- activation function: ReLU
- dropout: 0.2

4. **output :** 

- output : 1
- activation function: sigmoid



### Number of Parameters

![image-20210521162753359](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521162753359.png)

**The number of parameters: 3511 elements.**



### Weighted F1 Score I obtained

![image-20210521162905215](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521162905215.png)

### Hyper parameter setting

![image-20210521162946958](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521162946958.png)

mini-batch size : 1000

num_epoch : 500

loss function: torch.nn.BCELoss()

optimizer : torch.optim.SGD()
(learning_rate:0.01, weight decay: 0.02, momentum:0.95)

------

#### comments

unlike my conventional belief, larger mini-batch size such as 1000, yielded better f1 score. I`m tryint to figure out the reason. 



### Random Seeds Configuration

![image-20210521163147219](C:\Users\sodap\AppData\Roaming\Typora\typora-user-images\image-20210521163147219.png)

