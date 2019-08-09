from __future__ import division

import numpy as np

def to_landscape(diagram):
    """ Convert a diagram to a landscape
        (b,d) -> (b, d-b)
    """
    diagram[:, 1] -= diagram[:, 0]
    return diagram

def _consolidate(self, diagrams):
    """
    Consolidate all diagrams into one list.
    """
    return np.vstack(diagrams)

def _distanceclustercenter(self, point):
    """
    Return the index  of the closest cluster center from the given point.
    """
    return np.argmin([np.linalg.norm(point - x) for x in self.cluster_centers])

def weight_function(self, t):
    """
    Weight function proposed in the original paper to subsample
    """
    if t < self.a:
        return 0
    elif t >= self.a and t < self.b:
        return (t - self.a)/(self.b - self.a)
    else:
        return 1


def _gaussianweights(self, point, which_gaussian):
    """
    Parameter function defined in the original paper to calculate the significance of each
    point to the codeword.

    Usage
    ---------
    point : Persistence diagram point
    which_gaussian : int indicating which gaussian parameters should be used
    """
    # parameters of the gaussian and difference between mean and point
    mean_point = point - self.means_[which_gaussian]
    inverse_matrix = self.inverse_matrices_[which_gaussian]
    determinant_matrix = self.determinants_[which_gaussian]

    first_term = np.exp( -0.5 * np.dot(mean_point, np.matmul(inverse_matrix, mean_point)) )
    parameter_function = first_term/(2*np.pi*self.determinants_[which_gaussian])
    return parameter_function

def _sumithweights(self, landscape, which_gaussian):
    """
    Return the value of ith coordinate in the stable persistence bag of words for a specific
    landscape.
    """
    # select weight value to specified gaussian
    ith_weight = self.weights_[which_gaussian]
    # sum the contribution of all points in landscape to specified gaussian
    ith_value = np.sum(np.array([[_gaussianweights(self, point, which_gaussian)]
                                for point in landscape]))
    return ith_weight * ith_value

def _calcparameters(self):
    """
    Calculate the inverse and determinant from covariance matrices given in self.
    """
    determinants_ = []
    inverse_matrices_ = []
    for i in range(self.N):
        cov_matrix = self.covariances_[i]
        determinants_.append(np.linalg.det(cov_matrix))
        inverse_matrices_.append(np.linalg.inv(cov_matrix))
    # add to class
    self.determinants_ = np.array(determinants_)
    self.inverse_matrices_ = np.array(inverse_matrices_)

def _sumallweights(self, point):
    """
    Calculate the dot product between all gaussian values in a specific point and their respective
    weights
    """
    all_gaussian_weights = np.array([_gaussianweights(self, point, i) for i in range(self.N)])
    return np.dot(self.weights_, all_gaussian_weights)
