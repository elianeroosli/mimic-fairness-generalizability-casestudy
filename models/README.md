# Benchmark model training

Harutyunyan's paper looks at four clinical prediction tasks for ICU patients: 
in-hospital mortality, decompensation, length-of-stay and phenotyping. 
Five different baseline models for each of the four main tasks have been provided:

- Linear/logistic regression
- Standard LSTM
- Standard LSTM + deep supervision
- Channel-wise LSTM
- Channel-wise LSTM + deep supervision

In addition, they have also developped multitasking models that aim to learn all four
prediction tasks simultaneously. For the frame of this project however, we focus
on the modeling of in-hospital mortality. The best-performing
model for this task was reported to be the `simple channel-wise LSTM`. Hence, we focus
on analysing this specific model on bias, demographic fairness and generalizability.

# Benchmark model testing