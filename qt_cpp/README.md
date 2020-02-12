Implementation of various classification methods and approximate nearest neighbour algorithms with deep convolutional neural network (CNN).
Pre-requisites:
- Qt 5
- Opencv 3
- NonMetricSpaceLib

To run the application, setup the build dir to be located in the root folder of the project.

Probabilistic Neural Network with Orthogonal Series kernel is implemented in classification.cpp. If you use this code, please cite the paper Savchenko, A. (2017). Probabilistic Neural Network with Complex Exponential Activation Functions in Image Recognition using Deep Learning Framework. arXiv preprint arXiv:1708.02733.


Sequential three-way decisions based on distance ratio is implemented in ImageTesting.cpp. If you use this code, please cite the paper Savchenko, A. V. (2016). Fast multi-class recognition of piecewise regular objects based on sequential three-way decisions and granular computing. Knowledge-Based Systems, 91, 252-262.

Approximate nearest neighbour methods including maximal likelihood directed enumeration method are presented in ann.cpp. Their implementation in video face recognition task is shown in video.cpp. If you use this code, please cite the following papers:
- Savchenko, A. V. (2017). Maximum-likelihood approximate nearest neighbor method in real-time image recognition. Pattern Recognition, 61, 459-469.
- Savchenko, A. V. (2017). Deep neural networks and maximum likelihood search for approximate nearest neighbor in video-based image recognition. Optical Memory and Neural Networks, 26(2), 129-136.
- Savchenko, A. V. (2017). Deep Convolutional Neural Networks and Maximum-Likelihood Principle in Approximate Nearest Neighbor Search. In Iberian Conference on Pattern Recognition and Image Analysis (pp. 42-49). Springer, Cham.
