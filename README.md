# NN-CX

A simple neural network library developed from scratch to demonstrate popular machine learning usecases and for experimentation/learning purposes. `CX` represents the abstract compute backend design making it easier to add new backends and with the core design minimally changed. The library is under active development with new features added every week. 

### Features

- Native grad engine (inspired by micrograd)
- Native tensor class linked to a compute backend
- Neural network layers/operations built on top of tensor class
- Supporting blocks such as dataloader, datasets, models and trainers.


## Tensor backend support
- CPU via NumPy
- GPU via CuPy

## Apps
- Sine wave estimator (`apps/sine_pred.py`)
