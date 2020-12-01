# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training
```
# Start training with: 
CUDA_VISIBLE_DEVICES=0 python main.py

# You can manually resume the training with: 
CUDA_VISIBLE_DEVICES=0 python main.py --resume --lr=0.01
```
