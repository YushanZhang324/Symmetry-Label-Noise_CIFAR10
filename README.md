# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6.4
- PyTorch 0.4.0

## Training
```
# Start training with: 
CUDA_VISIBLE_DEVICES=0 python main.py

# You can manually resume the training with: 
CUDA_VISIBLE_DEVICES=0 python main.py --resume --lr=0.01
```
