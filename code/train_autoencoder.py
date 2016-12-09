import load_MNIST
import visualize
import utils
import gradient
import numpy
import scipy.optimize
import datetime
import os
import sys


# ======================================================================
# Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to achieve good filters;
#  you do not need to change the parameters below.

# number of input units
visible_size = 28 * 28




# desired average activation of the hidden units.
# (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
#  in the lecture notes).



# DEBUG (set to True in Ex 3)
DEBUG = False  # True

#Debug settings:
# number of hidden units
hidden_size = 2
# weight decay (weight regularization) parameter
# lambda_ = 0.

# Now that analytical gradient checks out to be correct when compared with numerical gradient estimate, we can add weight decay term:
lambda_ = 0.0001

problem = "d"

#rhoTest = .05
#rhoTest = .01
rhoTest = .005

if DEBUG is False:
    if problem is "a":
        hidden_size = 10
        betaTest = .2
    elif problem is "b":
        hidden_size = 50
        betaTest = .1
    elif problem is "c":
        hidden_size = 100 
        betaTest = .01   
    elif problem is "d":
        hidden_size = 250
        betaTest = .01



# ======================================================================
# Step 0: Load MNIST and visualize
# In this exercise, you will load the mnist dataset
# First download the dataset from the following website:
# Training Images: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
# Training Labels: http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz

# Loading Sample Images
# Loading 60K images from MNIST database

# NOTE: YOU LIKELY NEED TO CHANGE THE FOLLOWING PATH TO WHERE YOU STORED THE MNIST DATA
images = load_MNIST.load_MNIST_images('../data/train-images.idx3-ubyte')
# Each column represents one 28x28 pixel image (784 total pixels) that has
# been "unrolled" into a 784-element column vector.



# ======================================================================
# STEP 1: Visualize patches_train
# You will now use the visualize.py function plot_images() to display different
# sets if the MNIST dataset.  You can provide a filename argument to plot_images
# which will specify the name of the file the image is saved as;
# default is to save the file in the directory under the name 'weights.png'.
# Your task:
#     Plot the first 10, 50 and 100 images of patches_train
#     Also plot the first 100 image patches that will be used for 'testing' (patches 1200 to 1300)

# Code to visualize subsets of patches_train to commented out.
# Uncomment to run
#patches_train = images[:, 0:10]  # grabs the first 10 images (i.e., the first 10 columns)
#visualize.plot_images(patches_train[:, 0:100])

#patches_train = images[:, 0:50]  # grabs the first 50 images (i.e., the first 50 columns)
#visualize.plot_images(patches_train[:, 0:100])

patches_train = images[:, 0:100]  # grabs the first 100 images (i.e., the first 100 columns)
#visualize.plot_images(patches_train[:, 0:100])

#patches_train = images[:, 0:1000]  # grabs the first 1000 images (i.e., the first 100 columns)


patches_test = images[:, 1200:1300]  # grabs 100 image patches that will be used for 'testing'

#Visualize and save 100 patches from the test dataset
#visualize.plot_images(patches_test[:, 0:100], filepath = "../figures/100_test_patches.png")



# ======================================================================
# STEP 2: Implement utils.initialize
# Obtain random parameters theta ; see Exercise 2
# You need to implement initialize in utils.py
theta = utils.initialize(hidden_size, visible_size)


# ======================================================================
# STEP 3: Implement mlp_cost_and_grad
#
#  In this step you will implement the calculation of the loss (cost)
#  and theta gradient (grad) in utils.autoencoder_cost_and_grad.
#
#  You can implement all of the components in the cost function at once,
#  but it may be easier to do it step-by-step and run gradient checking
#  (see STEP 4) after each step.  We suggest implementing the
#  utils.autoencoder_cost_and_grad function using the following steps:
#
#  (a) Implement forward propagation in your neural network, and implement
#      the regularized loss function described in the tutorial and
#      class lectures.  Implement backpropagation to compute the derivatives.
#      Then (using lambda=beta=0), run Gradient Checking (i.e.,
#      gradient.compute_gradient_numerical_estimate) to verify that the
#      calculations of the gradient in utils.autoencoder_cost_and_grad
#      are sufficiently close to the numerical gradient estimate.
#
#  (b) Add in the weight decay term (in both the cost function and the
#      derivative calculations), then re-run Gradient Checking to verify
#      correctness.
#

#  Feel free to change the training settings when debugging your
#  code_solution.  For example, reducing the training set size and/or
#  the number of hidden units may make your code run faster; and
#  setting lambda_ to zero may be helpful for debugging.

(cost, grad) = utils.autoencoder_cost_and_grad(theta, visible_size, hidden_size, lambda_, patches_train)


# ======================================================================
# STEP 4: Gradient Checking
#
# Hint: If you are debugging your code_solution, performing gradient
# checking on smaller models and smaller training sets (e.g., using only
# 10 training examples and 1-2 hidden units) will speed things up.

# First, make sure your numerical gradient computation is correct
# for a simple function.

# After you have implemented gradient.compute_gradient_numerical_estimate,
# run this script while setting DEBUG = True on line 30 above.
# Once you are convinced your implementation of the cost and gradient
# calculation are correct, you can set DEBUG back to False (as when you
# return to realistic hidden unit and training data sizes, the gradient
# checking will get very slow!

if DEBUG:
    print "========== DEBUG: checking gradient =========="

    # the following test your implementation of compute_gradient_numerical_estimate
    gradient.test_compute_gradient_numerical_estimate()
    
    # Now we can use it to check your cost function and derivative calculations
    # for the sparse autoencoder.
    # J is the cost function

    print 'Now test autoencoder_cost_and_grad() gradient against numerical estimate:'
    print '    Total number of parameters, theta.shape= {0}'.format(theta.shape)

    # define the objective function that returns cost and grad, used by scipy.optimizze.minimize
    # NOTE NON-SPARSE VERSION OF autoencoder_cost_and_grad()!
    J = lambda x: utils.autoencoder_cost_and_grad(x, visible_size, hidden_size, lambda_, patches_train)
    num_grad = gradient.compute_gradient_numerical_estimate(J, theta)

    # The following is for optional additional debugging
    # Uncomment the following to look at the individual differences for each parameter.
    # Sometimes this can be informative.
    # for i in range(theta.shape[0]):
    #     if i == (visible_size * hidden_size) \
    #             or i == 2 * (visible_size * hidden_size)\
    #             or i == 2 * (visible_size * hidden_size) + hidden_size:
    #         print '======================='
    #     print '{0} : {1} {2} {3}'.format(i, num_grad[i] / 2 - grad[i], num_grad[i], grad[i])

    # Compare numerically computed gradients with the ones obtained from backpropagation
    diff = numpy.linalg.norm(num_grad - grad)
    print "Norm of the difference between numerical and autoencoder_cost_and_grad gradients:"
    print "    ", diff
    print "(should be at least < 1.0e-07)"

    print "========== DEBUG: checking gradient DONE =========="
"""
# ======================================================================
# STEP 5: After verifying that your implementation of
#  utils.autoencoder_cost_and_grad is correct, You can start training your
#  autoencoder, using scipy.optimize.minimize L-BFGS-B.

#  Initialize the parameters
theta = utils.initialize(hidden_size, visible_size)

print "\nRunning scipy.optimize.minimize on {0} parameters, over {1} training patches_train"\
    .format(theta.shape[0], patches_train.shape[1])
start_time = datetime.datetime.now()
print "    START TIME {0}".format(utils.get_pretty_time_string(start_time))
# define the objective function that returns cost and grad, used by scipy.optimizze.minimize
J = lambda x: utils.autoencoder_cost_and_grad(x, visible_size, hidden_size, lambda_, patches_train)
options_ = {'maxiter': 4000, 'disp': False}
result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
opt_theta = result.x  # theta found after optimization

end_time = datetime.datetime.now()
print "    END TIME {0}".format(utils.get_pretty_time_string(end_time))
total_time = end_time - start_time
time_elapsed_string = utils.get_pretty_time_string(total_time, delta=True)
print "    Total run time: {0}".format(time_elapsed_string)

print "\nscipy.optimize.minimize() details:"
print result

print "\nNOTE: Don't worry if scipy.optimize.minimize() reports 'success: False'"
print "      due to hitting the maximum iterations, or a message indicating an"
print "      error condition.  (nit = 'number of iterations')"
print "      I have set the max iterations to 4000, which is generally"
print "      sufficient for our purposes here."
"""
# ======================================================================
# SPARSE AUTOENCODER
#  After verifying that your implementation of
#  utils.autoencoder_cost_and_grad is correct, You can start training your
#  autoencoder, using scipy.optimize.minimize L-BFGS-B.

#  Initialize the parameters
theta = utils.initialize(hidden_size, visible_size)

print "\nRunning scipy.optimize.minimize on {0} parameters, over {1} training patches_train"\
    .format(theta.shape[0], patches_train.shape[1])
start_time = datetime.datetime.now()
print "    START TIME {0}".format(utils.get_pretty_time_string(start_time))
# define the objective function that returns cost and grad, used by scipy.optimizze.minimize
# autoencoder_cost_and_grad_sparse(theta, visible_size, hidden_size, lambda_, rho_, beta_, data)
J = lambda x: utils.autoencoder_cost_and_grad_sparse(x, visible_size, hidden_size, lambda_, rhoTest, betaTest, patches_train)
options_ = {'maxiter': 4000, 'disp': False}
result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options_)
opt_theta = result.x  # theta found after optimization

end_time = datetime.datetime.now()
print "    END TIME {0}".format(utils.get_pretty_time_string(end_time))
total_time = end_time - start_time
time_elapsed_string = utils.get_pretty_time_string(total_time, delta=True)
print "    Total run time: {0}".format(time_elapsed_string)

print "\nscipy.optimize.minimize() details:"
print result

print "\nNOTE: Don't worry if scipy.optimize.minimize() reports 'success: False'"
print "      due to hitting the maximum iterations, or a message indicating an"
print "      error condition.  (nit = 'number of iterations')"
print "      I have set the max iterations to 4000, which is generally"
print "      sufficient for our purposes here."


# ======================================================================
# STEP 6: Visualize and save results

results_filepath_root = 'autoencoder_k{0}_h{1}_l{2}'.format(patches_train.shape[1], hidden_size, "point0001")

utils.plot_and_save_results\
    (opt_theta, visible_size, hidden_size,
     root_filepath=results_filepath_root,
     train_patches=patches_train[:, 0:100],  # be sure to only use the first 100 patches, or visualization may get messy
     test_patches=patches_test,
     show_p=False,
     # Everything after this point is stored as a dictionary in the 'params' argument:
     lambda_=lambda_,
     # rho_=rho_,    # for when you implement utils.autoencoder_cost_and_grad_sparse
     # beta_=beta_,  # for when you implement utils.autoencoder_cost_and_grad_sparse
     train_time=time_elapsed_string,
     nit=result.nit,
     success=result.success,
     message=result.message)


#####
# Stop execution here...
sys.exit()
# Move the above line to different parts of the assignment
#   as you implement more of the functionality.
#####

