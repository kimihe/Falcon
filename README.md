# Falcon
> All the practice of Falcon is for project Griffin.

A computation-parallel deep learning architecture.

# Quick Start of SSP Demo
The SSP demo is a part of Falcon, please find it in the directory `SSP_Demo` (all the demo files and related dataset are in this directory).

We hope to this can help researchers to start their first distrtibuted DL training in SSP scheme via [PyTorch](https://pytorch.org/).

The implementation guideline follows the papers as:

* Q. Ho, J. Cipar, H. Cui, J. K. Kim, S. Lee, P. B. Gibbons, G. A. Gibson, G. R. Ganger, and E. P. Xing, "[More effective distributed ml via a stale synchronous parallel parameter server](https://dl.acm.org/citation.cfm?id=2999748)," in *Proc. NIPS*, Lake Tahoe, Nevada, USA, 2013.
* W. Zhang, S. Gupta, X. Lian, and J. Liu, "[Staleness-aware async-sgd for distributed deep learning](https://dl.acm.org/citation.cfm?id=3060832.3060950)," in *Proc. IJCAI*, New York, USA, 2016.

# Dataset
Two classical datasets are supported: MNIST and CIFAR-10.

* MNIST: This demo has already contained MNIST in the directory `data`, you can also download it from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* CIFAR-10: You can download it from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

