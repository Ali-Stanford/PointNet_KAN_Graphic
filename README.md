# PointNet with KAN versus PointNet with MLP for 3D Classification and Segmentation of Point Sets

![pic](./network.png)

**Abstract** <be>

We introduce PointNet-KAN, a neural network for 3D point cloud classification and segmentation tasks, built upon two key components. First, it employs Kolmogorov-Arnold Networks (KANs) instead of traditional Multilayer Perceptrons (MLPs). Second, it retains the core principle of PointNet by using shared KAN layers and applying symmetric functions for global feature extraction, ensuring permutation invariance with respect to the input features. In traditional MLPs, the goal is to train the weights and biases with fixed activation functions; however, in KANs, the goal is to train the activation functions themselves. We use Jacobi polynomials to construct the KAN layers. We extensively evaluate PointNet-KAN across various polynomial degrees and special types such as the Lagrange, Chebyshev, and Gegenbauer polynomials. Our results show that PointNet-KAN achieves competitive performance compared to PointNet with MLPs on benchmark datasets for 3D object classification and segmentation, despite employing a shallower and simpler network architecture. We hope this work serves as a foundation and provides guidance for integrating KANs, as an alternative to MLPs, into more advanced point cloud processing architectures.

**Installation** <br>
