# Aggregated Momentum

This repository contains code to reproduce the experiments from ["Aggregated Momentum: Stability Through Passive Damping"](https://arxiv.org/abs/1804.00325).

A pytorch implementation of AggMo can be found in `aggmo.py` .

## AggMo Optimizer

### Pytorch

The `aggmo.py` file provides a pytorch implementation. The optimizer can be constructed as follows:

```python
optimizer = aggmo.AggMo(model.parameters(), lr, betas=[0, 0.9, 0.99])
```

The AggMo class also has an "exponential form" constructor. In this case the damping vector is specified by two hyparameters, `K` - the number of beta values, and `a` the scale factor. For i=1...K each beta_i = 1 - a^i .
The following is equivalent to the optimizer with momentum [0, 0.9, 0.99]:

```python
optimizer = aggmo.AggMo.from_exp_form(model.parameters(), lr, a=0.1, K=3)
```

### Tensorflow

There is also a tensorflow implementation within the `tensorflow` folder. **This version has not been carefully tested**.

```python
optimizer = aggmo.AggMo(lr, betas=[0, 0.9, 0.99])
```

Or using the exponential form:

```python
optimizer = aggmo.AggMo.from_exp_form(lr, a=0.1, K=3)
```

## Running Experiments

All experiments can be run using the same entry point. Each task and optimizer has their own config file which can be easily overridden from the command line.

The first argument points to the task configuration. The optimizer is specified with `--optim <optimizer_name>`. Additional config overrides can be given after `-o` in the format e.g. `-o optim.lr_schedule.lr_decay=0.5`.

The optimizer configs will not provide optimal hyperparameters for every task.


### Autoencoders

From the `src` directory:

```python
python main.py configs/ae.json --optim aggmo
```

### Classification

From the `src` directory:

```python
python main.py configs/cifar-10.json --optim aggmo
```


```python
python main.py configs/cifar-100.json --optim aggmo
```

### LSTMs

The LSTM code is not directly included here. We made direct use of the [official code](https://github.com/salesforce/awd-lstm-lm) from ["Regularizing and Optimizing LSTM Language Models"](https://arxiv.org/abs/1708.02182). You can run these experiments by using the AggMo optimizer within this repository. The model hyperparameters used are detailed in the appendix.