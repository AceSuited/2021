# HW#4 Spec

## HW#4 - Deep Neural Networks

Hi,

Please use the attached train and test files. You already have used this format of data for previous assignments. I think you can easily understand the format. X contains data points and Y contains their class labels.

\1. Use the training data to train your model and test with the testing data.

\2. Design your own model.

\3. You should output i) the number of parameters, and ii) the weighted F-1 score. Please refer to the following figure which contains an example about how to do it. My model has 6351 elements (parameters) and marked a weighted F-1 of 0.8248.

\4. You cannot modify the data (e.g., applying the poly kernel is not allowed) but focus on improving the performance only by designing your own neural networks. You can use any techniques you are aware of (e.g., regularization, early stopping of training, learning rate scheduling, batch norm, etc.)

\5. The grading will be done by finding Pareto-frontiers. For this, I will project your submission onto the 2-dimensional space of (model size, weighted F-1) and find Pareto-frontiers. 

\6. I will ignore the three least significant values of the model size, i.e., 6351 will be considered as 6000. In fact, X000 ~ X999 will be considered as X000.

\7. I will consider up to the fourth decimal point of the weighted F-1, i.e., 0.82481868 will be considered as 0.8248.

\8. Note that in 6 and 7, I used the rounding down. Be careful about this. You should report the full precision values and TAs will convert your reported values as mentioned above.

\9. In my example codes, I do not use stochastic gradient descent but train with the entire training data. You should change this to use stochastic gradient descent with many mini-batches. You can define your mini-batch size.

\10. In my example codes, I use the checkpoint at the last epoch, which may be sub-optimal. You should implement your model selection procedure, e.g., using the checkpoint with the minimum training loss and using the cross-validation with the training data.

\11. 0 pts if at least one of the above requirements is not met. Note that 9 and 10 are mandatory. If you do not implement them, 0 pts will be given.



\12. The first Pareto-frontiers will have 100 pts, the second 90 pts, the third 80 pts, and so on.



Submit your Jupyter Notebook and a report describing your design, the number of parameters, and the weighted F-1 you obtained.