# Authors: Sean Hendryx, Dr. Clayton Morrison
# Written for Machine Learning, INFO 521, at the University of Arizona
# December 2016
# Code adapted from Stanford's: http://ufldl.stanford.edu/wiki/index.php/Neural_Networks, and ideas from https://github.com/jatinshah/ufldl_tutorial

# Utilities for neural net

# References:
# http://ufldl.stanford.edu/wiki/index.php/Neural_Networks
# http://neuralnetworksanddeeplearning.com/chap1.html
# https://github.com/jatinshah/ufldl_tutorial
# https://docs.scipy.org
# Python machine learning by Raschka, Sebastian
# http://stackoverflow.com/questions/34912658/trying-to-understand-gradient-checking-error-in-3-layer-neural-network


import numpy
import math
import os
import visualize
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoidPrime(z):
    #Note that this function takes in z values, i.e. the weighted sum of activations, a, + b values
    # by mapping z through sigmoid(), we get: a(1-a)
    return sigmoid(z) * (1 - sigmoid(z))

# -------------------------------------------------------------------------

def initialize(hidden_size, visible_size):
    """
    Sample weights uniformly from the interval [-r, r] as described in lecture 23.
    Return 1d array theta (in format as described in Exercise 2)
    :param hidden_size: number of hidden units
    :param visible_size: number of visible units (of input and output layers of autoencoder)
    :return: theta array
    """

    ### YOUR CODE HERE ###
    # Define lower and upper bounds from initialization heuristic
    # n_in = visible_size
    # and n_out = visible_size, since we are training an autoencoder
    lower = -1. * numpy.sqrt(6./(visible_size + visible_size + 1.))
    upper = numpy.sqrt(6./(visible_size + visible_size + 1.))
    
    # W^{(l)}_{ij} denotes the parameter (or weight) associated with the connection between unit j in layer l, and unit i in layer l + 1. (Note the order of the indices.)
    # ROWS INDICATE "TO" NODES AND COLUMNS INDICATE "FROM" NODES
    W1 = numpy.random.uniform(lower, upper, size = (hidden_size, visible_size))
    W2 = numpy.random.uniform(lower, upper, size = (visible_size, hidden_size))

    # b^{(l)}_i is the bias associated with unit i in layer l + 1
    b1 = numpy.random.uniform(lower, upper, size = hidden_size)
    b2 = numpy.random.uniform(lower, upper, size = visible_size)

    theta = numpy.concatenate((numpy.reshape(W1,W1.size),numpy.reshape(W2,W2.size), numpy.reshape(b1,b1.size), numpy.reshape(b2,b2.size)))

    return theta


# -------------------------------------------------------------------------

def autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, data):
    """
    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """
    
    ### YOUR CODE HERE ###
    
    # theta is an array with order [{W(1)}, {W(2)}, {b(1)}, {b(2)}]
    # in W, ROWS INDICATE "TO" NODES AND COLUMNS INDICATE "FROM" NODES
    # Pull values from theta vector and reshape:
    W1 = theta[0:(hidden_size * visible_size)]
    W1 = numpy.reshape(W1, (hidden_size, visible_size))
    
    W2 = theta[(hidden_size * visible_size):((hidden_size * visible_size) + (visible_size * hidden_size))]
    W2 = numpy.reshape(W2, (visible_size, hidden_size))
    
    b1 = theta[((hidden_size * visible_size) + (visible_size * hidden_size)):(((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size)]
    b2 = theta[(((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size) : (((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size + visible_size)]
    
    ##########################################################################################################################################
    # FEED FORWARD/FORWARD PROPOGATION:
    # in W, ROWS INDICATE "TO" NODES (i) AND COLUMNS INDICATE "FROM" NODES (j)
    # Activations at layer 1 = inputs, i.e., aSup1 = x
    # Number of neurons = number of input data points (pixels), e.g. 784, which we can also say is the visible size?
    
    # In the sequel, we also let z^{(l)}_i denote the total weighted sum of inputs to unit i in layer l, including the bias term (e.g., \textstyle z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i), so that a^{(l)}_i = f(z^{(l)}_i).
    # http://ufldl.stanford.edu/wiki/index.php/Neural_Networks
    
    # Number of training points
    m = data.shape[1]
    
    # note that activations at the first layer are equal to the input data:
    #    a_i^{(1)} = x_i
    # Compute z values at second layer
    # zSup2 (i.e., z^{(2)}) is the matrix of z values at layer 2
    # zSup2 = W^{(1)} x + b^{(1)}
    zSup2 = W1.dot(data) + numpy.tile(b1, (m, 1)).transpose()
    
    # Compute activations at second layer by mapping z^{(2)} to sigmoid(z^{(2)})
    aSup2 = sigmoid(zSup2)
    
    #Compute z at third layer, z^{(3)}
    zSup3 = W2.dot(aSup2) + numpy.tile(b2, (m, 1)).transpose()
    # z at third layer is the total weighted sum of inputs to unit i in layer 3,
    # hypothesis = activation at the third layer: hypothesis = f(z^{(3)})
    hypothesis = sigmoid(zSup3)
    
    ##########################################################################################################################################
    # COMPUTE COST
    
    # Now add weight decay term with lambda_:
    #here
    # to sum over i and j in summation loops, we can use numpy.sum of W1 and W2 and then add the two summations together to account for the outermost summation to sum over all layers - 1
    cost = numpy.sum((hypothesis - data) ** 2.) / (2. * m) + (lambda_ / 2.) * ( numpy.sum(W1 **2) + numpy.sum(W2 ** 2) )
    
    #TRIED WITH numpy.linalg.norm() and found it to be twice as slow as above implementation of cost:
    #start = time.clock()
    #for n in range(40000):
    #	costNorm = (1./(2. * m)) * numpy.linalg.norm(numpy.dstack((hypothesis, data)))**2
    
    #print time.clock() - start
    # 5.894494
    
    #Compared to:
    #for n in range(40000):
    #	costNorm = (1./(2. * m)) * numpy.linalg.norm(numpy.dstack((hypothesis, data)))**2
    
    #print time.clock() - start
    #2.99788
    
    ##########################################################################################################################################
    # BACK PROPOGATION
    # Compute deltas:
    
    #\delta^{(3)}, i.e. output layer
    deltaSup3 = -1. * (data - hypothesis) * sigmoidPrime(zSup3)
    
    #\delta^{(2)}, i.e. hidden layer
    deltaSup2 = numpy.dot(W2.transpose(), deltaSup3) * sigmoidPrime(zSup2)
    
    ##########################################################################################################################################
    # Compute gradients:
    
    # working "backwards" from output to input
    grad_WSup2 = ((1.0/m) * numpy.dot(deltaSup3, aSup2.transpose())) + (lambda_ * W2)
    
    #or with numpy.outer:
    #Onabla_WSup2 = numpy.outer(deltaSup3, aSup2)
    # ^ dont think this is right
    
    grad_WSup1 = ((1.0/m) * numpy.dot(deltaSup2, data.transpose())) + lambda_ * W1
    grad_WSup1_2 = deltaSup2.dot(data.transpose()) / m + lambda_ * W1
    
    
    grad_bSup2 = (1.0/m) * numpy.sum(deltaSup3, axis = 1)
    
    grad_bSup1 = (1.0/m) * numpy.sum(deltaSup2, axis = 1)
    
    grad = numpy.concatenate((numpy.reshape(grad_WSup1,W1.size), numpy.reshape(grad_WSup2,W2.size), numpy.reshape(grad_bSup1,b1.size), numpy.reshape(grad_bSup2,b2.size)))
    
    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, data):
    """
    Version of cost and grad that incorporates sparsity constraint
        rho_ : the target sparsity limit for each hidden node activation
        beta_ : controls the weight of the sparsity pentalty term relative
                to other loss components

    The input theta is a 1-dimensional array because scipy.optimize.minimize expects
    the parameters being optimized to be a 1d array.
    First convert theta from a 1d array to the (W1, W2, b1, b2)
    matrix/vector format, so that this follows the notation convention of the
    lecture notes and tutorial.
    You must compute the:
        cost : scalar representing the overall cost J(theta)
        grad : array representing the corresponding gradient of each element of theta
    """

    ### YOUR CODE HERE ###
    # theta is an array with order [{W(1)}, {W(2)}, {b(1)}, {b(2)}]
    # in W, ROWS INDICATE "TO" NODES AND COLUMNS INDICATE "FROM" NODES
    # Pull values from theta vector and reshape:
    W1 = theta[0:(hidden_size * visible_size)]
    W1 = numpy.reshape(W1, (hidden_size, visible_size))
    
    W2 = theta[(hidden_size * visible_size):((hidden_size * visible_size) + (visible_size * hidden_size))]
    W2 = numpy.reshape(W2, (visible_size, hidden_size))
    
    b1 = theta[((hidden_size * visible_size) + (visible_size * hidden_size)):(((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size)]
    b2 = theta[(((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size) : (((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size + visible_size)]
    
    ##########################################################################################################################################
    # FEED FORWARD/FORWARD PROPOGATION:
    # in W, ROWS INDICATE "TO" NODES (i) AND COLUMNS INDICATE "FROM" NODES (j)
    # Activations at layer 1 = inputs, i.e., aSup1 = x
    # Number of neurons = number of input data points (pixels), e.g. 784, which we can also say is the visible size?
    
    # In the sequel, we also let z^{(l)}_i denote the total weighted sum of inputs to unit i in layer l, including the bias term (e.g., \textstyle z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i), so that a^{(l)}_i = f(z^{(l)}_i).
    # http://ufldl.stanford.edu/wiki/index.php/Neural_Networks
    
    # Number of training points
    m = data.shape[1]
    
    # note that activations at the first layer are equal to the input data:
    #    a_i^{(1)} = x_i
    # Compute z values at second layer
    # zSup2 (i.e., z^{(2)}) is the matrix of z values at layer 2
    # zSup2 = W^{(1)} x + b^{(1)}
    zSup2 = W1.dot(data) + numpy.tile(b1, (m, 1)).transpose()
    
    # Compute activations at second layer by mapping z^{(2)} to sigmoid(z^{(2)})
    aSup2 = sigmoid(zSup2)
    
    #Compute z at third layer, z^{(3)}
    zSup3 = W2.dot(aSup2) + numpy.tile(b2, (m, 1)).transpose()
    # z at third layer is the total weighted sum of inputs to unit i in layer 3,
    # hypothesis = activation at the third layer: hypothesis = f(z^{(3)})
    hypothesis = sigmoid(zSup3)
    
    ##########################################################################################################################################
    # COMPUTE COST
    
    # Now add sparsity (computed from activations to the output layer):
    rhoHat = numpy.sum(aSup2, axis=1)/m
    # Turn rho_ into matrix for vectorized computation
    rho = numpy.tile(rho_, hidden_size)
    
    # to sum over i and j in summation loops, we can use numpy.sum of W1 and W2 and then add the two summations together to account for the outermost summation to sum over all layers - 1
    # now with sparsity implemented: beta_ parameter determines amount of penalty applied relative to the regular cost function (smaller beta = less penalty)
    # Extra penalty term to the optimization objective penalizes rhoHat for deviating significantly from rho
    cost = numpy.sum((hypothesis - data) ** 2.) / (2. * m) + (lambda_ / 2.) * ( numpy.sum(W1 **2) + numpy.sum(W2 ** 2) ) + beta_ * numpy.sum(rho * numpy.log(rho / rhoHat) + ((1 - rho) * numpy.log((1 - rho) / (1 - rhoHat))))
    

    

    #TRIED WITH numpy.linalg.norm() and found it to be twice as slow as above implementation of cost:
    #start = time.clock()
    #for n in range(40000):
    #   costNorm = (1./(2. * m)) * numpy.linalg.norm(numpy.dstack((hypothesis, data)))**2
    
    #print time.clock() - start
    # 5.894494
    
    #Compared to:
    #for n in range(40000):
    #   costNorm = (1./(2. * m)) * numpy.linalg.norm(numpy.dstack((hypothesis, data)))**2
    
    #print time.clock() - start
    #2.99788
    
    ##########################################################################################################################################
    # BACK PROPOGATION
    # Compute deltas:
    
    #\delta^{(3)}, i.e. output layer
    deltaSup3 = -1. * (data - hypothesis) * sigmoidPrime(zSup3)
    
    #\delta^{(2)}, i.e. hidden layer
    # Use numpy.tile to vectorize computation by tiling out m training examples
    deltaSup2 = (numpy.dot(W2.transpose(), deltaSup3) + beta_ * (numpy.tile((-1. * rho / rhoHat) + ( (1 - rho) / (1 - rhoHat) ), (m, 1)).transpose()) ) * sigmoidPrime(zSup2)
    
    ##########################################################################################################################################
    # Compute gradients:
    
    # working "backwards" from output to input
    grad_WSup2 = ((1.0/m) * numpy.dot(deltaSup3, aSup2.transpose())) + (lambda_ * W2)
    
    #or with numpy.outer:
    #Onabla_WSup2 = numpy.outer(deltaSup3, aSup2)
    # ^ dont think this is right
    
    grad_WSup1 = ((1.0/m) * numpy.dot(deltaSup2, data.transpose())) + lambda_ * W1
    grad_WSup1_2 = deltaSup2.dot(data.transpose()) / m + lambda_ * W1
    
    
    grad_bSup2 = (1.0/m) * numpy.sum(deltaSup3, axis = 1)
    
    grad_bSup1 = (1.0/m) * numpy.sum(deltaSup2, axis = 1)
    
    grad = numpy.concatenate((numpy.reshape(grad_WSup1,W1.size), numpy.reshape(grad_WSup2,W2.size), numpy.reshape(grad_bSup1,b1.size), numpy.reshape(grad_bSup2,b2.size)))
    


    return cost, grad


# -------------------------------------------------------------------------

def autoencoder_feedforward(theta, visible_size, hidden_size, data):
    """
    Given a definition of an autoencoder (including the size of the hidden
    and visible layers and the theta parameters) and an input data matrix
    (each column is an image patch, with 1 or more columns), compute
    the feedforward activation for the output visible layer for each
    data column, and return an output activation matrix (same format
    as the data matrix: each column is an output activation "image"
    corresponding to the data input).

    Once you have implemented the autoencoder_cost_and_grad() function,
    simply copy your initial codes that computes the feedforward activations
    up to the output visible layer activations and return that activation.
    You do not need to include any of the computation of cost or gradient.
    It is likely that your implementation of feedforward in your
    autoencoder_cost_and_grad() is set up to handle multiple data inputs,
    in which case your only task is to ensure the output_activations matrix
    is in the same corresponding format as the input data matrix, where
    each output column is the activation corresponding to the input column
    of the same column index.

    :param theta: the parameters of the autoencoder, assumed to be in this format:
                  { W1, W2, b1, b2 }
                  W1 = weights of layer 1 (input to hidden)
                  W2 = weights of layer 2 (hidden to output)
                  b1 = layer 1 bias weights (to hidden layer)
                  b2 = layer 2 bias weights (to output layer)
    :param visible_size: number of nodes in the visible layer(s) (input and output)
    :param hidden_size: number of nodes in the hidden layer
    :param data: input data matrix, where each column is an image patch,
                  with one or more columns
    :return: output_activations: an matrix output, where each column is the
                  vector of activations corresponding to the input data columns
    """

    ### YOUR CODE HERE ###
    # theta is an array with order [{W(1)}, {W(2)}, {b(1)}, {b(2)}]
    # in W, ROWS INDICATE "TO" NODES AND COLUMNS INDICATE "FROM" NODES
    # Pull values from theta vector and reshape:
    W1 = theta[0:(hidden_size * visible_size)]
    W1 = numpy.reshape(W1, (hidden_size, visible_size))
    
    W2 = theta[(hidden_size * visible_size):((hidden_size * visible_size) + (visible_size * hidden_size))]
    W2 = numpy.reshape(W2, (visible_size, hidden_size))
    
    b1 = theta[((hidden_size * visible_size) + (visible_size * hidden_size)):(((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size)]
    b2 = theta[(((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size) : (((hidden_size * visible_size) + (visible_size * hidden_size)) + hidden_size + visible_size)]
    
    ##########################################################################################################################################
    # FEED FORWARD/FORWARD PROPOGATION:
    # in W, ROWS INDICATE "TO" NODES (i) AND COLUMNS INDICATE "FROM" NODES (j)
    # Activations at layer 1 = inputs, i.e., aSup1 = x
    # Number of neurons = number of input data points (pixels), e.g. 784, which we can also say is the visible size?
    
    # In the sequel, we also let z^{(l)}_i denote the total weighted sum of inputs to unit i in layer l, including the bias term (e.g., \textstyle z_i^{(2)} = \sum_{j=1}^n W^{(1)}_{ij} x_j + b^{(1)}_i), so that a^{(l)}_i = f(z^{(l)}_i).
    # http://ufldl.stanford.edu/wiki/index.php/Neural_Networks
    
    # Number of training points
    m = data.shape[1]
    
    # note that activations at the first layer are equal to the input data:
    #    a_i^{(1)} = x_i
    # Compute z values at second layer
    # zSup2 (i.e., z^{(2)}) is the matrix of z values at layer 2
    # zSup2 = W^{(1)} x + b^{(1)}
    zSup2 = W1.dot(data) + numpy.tile(b1, (m, 1)).transpose()
    
    # Compute activations at second layer by mapping z^{(2)} to sigmoid(z^{(2)})
    aSup2 = sigmoid(zSup2)
    
    #Compute z at third layer, z^{(3)}
    zSup3 = W2.dot(aSup2) + numpy.tile(b2, (m, 1)).transpose()
    # z at third layer is the total weighted sum of inputs to unit i in layer 3,
    # hypothesis = activation at the third layer: hypothesis = f(z^{(3)})
    output_activations = sigmoid(zSup3)
    
    return output_activations


# -------------------------------------------------------------------------

def save_model(theta, visible_size, hidden_size, filepath, **params):
    numpy.savetxt(filepath + '_theta.csv', theta, delimiter=',')
    with open(filepath + '_params.txt', 'a') as fout:
        params['visible_size'] = visible_size
        params['hidden_size'] = hidden_size
        fout.write('{0}'.format(params))


# -------------------------------------------------------------------------

def plot_and_save_results(theta, visible_size, hidden_size, root_filepath=None,
                          train_patches=None, test_patches=None, show_p=False,
                          **params):
    """
    This is a helper function to streamline saving the results of an autoencoder.
    The visible_size and hidden_size provide the information needed to retrieve
    the autoencoder parameters (w1, w2, b1, b2) from theta.

    This function does the following:
    (1) Saves the parameters theta, visible_size and hidden_size as a text file
        called '<root_filepath>_model.txt'
    (2) Extracts the layer 1 (input-to-hidden) weights and plots them as an image
        and saves the image to file '<root_filepath>_weights.png'
    (3) [optional] train_patches are intended to be a set of patches that were
        used during training of the autoencoder.  Typically these will be the first
        100 patches of the MNIST data set.
        If provided, the patches will be given as input to the autoencoder in
        order to generate output 'decoded' activations that are then plotted as
        patches in an image.  The image is saved to '<root_filepath>_train_decode.png'
    (4) [optional] test_patches are intended to be a different set of patches
        that were *not* used during training.  This permits inspecting how the
        autoencoder does decoding images it was not trained on.  The output activation
        image is generated the same way as in step (3).  The image is save to
        '<root_filepath>_test_decode.png'

    If train_patches are provided, will compute the feedforward
        activation of the

    The root_filepath is used as the base filepath name for all files generated
    by this function.  For example, if you wish to save all of the results
    using the root_filepath='results1', and you have specified the train_patches and
    test_patches, then the following files will be generated:
        results1_model.txt
        results1_weights.png
        results1_train_decode.png
        results1_test_decode.png
    If no root_filepath is provided, then the output will default to:
        model.txt
        weights.png
        train_decode.png
        test_decode.png
    Note that if those files already existed, they will be overwritten.

    :param theta: model parameters
    :param visible_size: number of nodes in autoencoder visible layer
    :param hidden_size: number of nodes in autoencoder hidden layer
    :param root_filepath: base filepath name for files generated by this function
    :param train_patches: matrix of patches (typically the first 100 patches of MNIST)
    :param test_patches: matrix of patches (intended to be patches NOT used in training)
    :param show_p: flag specifying whether to show the plots before exiting
    :param params: optional parameters that will be saved with the model as a dictionary
    :return:
    """

    filepath = 'model'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    save_model(theta, visible_size, hidden_size, filepath, **params)

    # extract the input to hidden layer weights and visualize all the weights
    # corresponding to each hidden node
    w1 = theta[0:hidden_size * visible_size].reshape(hidden_size, visible_size).T
    filepath = 'weights.png'
    if root_filepath:
        filepath = root_filepath + '_' + filepath
    visualize.plot_images(w1, show_p=False, filepath=filepath)

    if train_patches is not None:
        # Given: train_patches and autoencoder parameters,
        # compute the output activations for each input, and plot the resulting decoded
        # output patches in a grid.
        # You can then manually compare them (visually) to the original input train_patches
        filepath = 'train_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        train_decode = autoencoder_feedforward(theta, visible_size, hidden_size, train_patches)
        visualize.plot_images(train_decode, show_p=False, filepath=filepath)

    if test_patches is not None:
        # Same as for train_patches, but assuming test_patches are patches that were not
        # used for training the autoencoder.
        # Again, you can then manually compare the decoded patches to the test_patches
        # given as input.
        test_decode = autoencoder_feedforward(theta, visible_size, hidden_size, test_patches)
        filepath = 'test_decode.png'
        if root_filepath:
            filepath = root_filepath + '_' + filepath
        visualize.plot_images(test_decode, show_p=False, filepath=filepath)

    if show_p:
        plt.show()


# -------------------------------------------------------------------------

def get_pretty_time_string(t, delta=False):
    """
    Really cheesy kludge for producing semi-human-readable string representation of time
    y = Year, m = Month, d = Day, h = Hour, m (2nd) = minute, s = second, mu = microsecond
    :param t: datetime object
    :param delta: flag indicating whether t is a timedelta object
    :return:
    """
    if delta:
        days = t.days
        hours = t.seconds // 3600
        minutes = (t.seconds // 60) % 60
        seconds = t.seconds - (minutes * 60)
        return 'days={days:02d}, h={hour:02d}, m={minute:02d}, s={second:02d}' \
                .format(days=days, hour=hours, minute=minutes, second=seconds)
    else:
        return 'y={year:04d},m={month:02d},d={day:02d},h={hour:02d},m={minute:02d},s={second:02d},mu={micro:06d}' \
                .format(year=t.year, month=t.month, day=t.day,
                        hour=t.hour, minute=t.minute, second=t.second,
                        micro=t.microsecond)


# -------------------------------------------------------------------------
