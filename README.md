# Wav2vec 2.0
Foreigner Korean speech voice recognition hackathon - CSLEE

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip3 install --editable ./
python3 setup.py build develop
```
- We only test this implementation in Ubuntu 20.04.

