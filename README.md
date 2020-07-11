# Implementation of "Matrix Capsules with EM Routing"

A TensorFlow implementation of Hinton's paper [Matrix Capsules with EM Routing](https://openreview.net/pdf?id=HJWLfGWRb) by [Perry Deng](https://github.com/PerryXDeng).

E-mail: [perry.deng@mail.rit.edu](mailto:perry.deng@mail.rit.edu)

This implementation experiments with modifications to Hinton's implementation, the main ones being:

1. affine instead of linear vote calculation
2. dropout and dropconnect capsule layers
3. detection of adversarial patches using reconstruction networks

# Acknowledgements

1. [Jonathan Hui's blog](https://jhui.github.io/2017/11/14/Matrix-Capsules-with-EM-routing-Capsule-Network/), "Understanding Matrix capsules with EM Routing (Based on Hinton's Capsule Networks)"
2. [Questions and answers](https://openreview.net/forum?id=HJWLfGWRb) on OpenReview, "Matrix capsules with EM routing"
3. [Suofei Zhang's implementation](https://github.com/www0wwwjs1/Matrix-Capsules-EM-Tensorflow) on GitHub, "Matrix-Capsules-EM-Tensorflow" 
4. [Guang Yang's implementation](https://github.com/gyang274/capsulesEM) on GitHub, "CapsulesEM"
5. [A. Gritzman's implementation](https://arxiv.org/pdf/1907.00652.pdf)
