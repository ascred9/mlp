# Multi Layer Perceptron
My own project to implement the neural networks to explore ideas and target use with more flexibility.
Below I will try to get the small description how to use this mlp.
Well, the class Network is a manager of LayerDeque class. Network read, write the optional files, start the testing and training and so on.
LayerDeque presents the communication between each Layer, controls the training, saves temporal variables.
Layer consists on the matrix with wieghts and also the activation function. At this moment two types of networks are implemented: Standard and Bayesian.