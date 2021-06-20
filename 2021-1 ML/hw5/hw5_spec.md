## Final Homework: Time-series Prediction

Hi Everyone,

I announce the final assignment for time-series prediction. The train/test sets look as follows: i) read 5 recent historical values and predict the next 5 values (i.e., sequence-to-sequence prediction), ii) 15591 training samples exist and note that [:,:,0] contains the 5 historical values, and [:,:,1] contains the next 5 values to predict, iii) the same data format is used for the test set.

You have to use the same protocol that we used for HW4, e.g., validation, grading by Pareto-frontiers, etc. You can use LSTM, GRU, 1D CNN, or Fully connected layers for this task. Please find the most efficient architecture for the given time-series dataset. Please submit your Jupyter notebook in Google Colab and a report summarizing your model architecture/model size/model accuracy in terms of the mean absolute percentage error (MAPE) in https://en.wikipedia.org/wiki/Mean_absolute_percentage_error.

The attached datasets are in the NumPy format and you can easily read them.

