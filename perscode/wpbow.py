from __future__ import division
import collections

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans

from .utils import to_landscape, _consolidate, _distanceclustercenter, weight_function

__all__ = ['wPBoW']

class wPBoW(TransformerMixin):
    """ Initialize a Persistence Bag of Words generator.


    Parameters
    -----------
    N : Length of codebook (length of vector)
    n_subsample : Number of points to be subsampled when calculating GMMs
    normalize : if set to True normalizes each vectorized persistence diagram with respect
                to the euclidean norm
    cluster_centers : list or numpy array of 2d vectors representing the cluster centers. The
                      length of the list should be equal to N.

    Usage
    -----------
    >>> import perscode
    >>> # define length of codebook
    >>> length_codebook = 10
    >>> # number of points to be subsampled
    >>> n_subsample = 10
    >>> wpbow = perscode.wPBoW(N = length_codebook, n_subsample = 10)
    >>> wpbow_diagrams = wpbow.transform(diagrams) # diagrams is a list of persistence diagrams

    """
    def __init__(
        self,
        N = 10,
        n_subsample = 100,
        normalize = False,
        cluster_centers = None,
        ):
        # size of codebook
        self.N = N
        # number of points to be subsampled from the consolidated persistence diagram
        self.n_subsample = n_subsample
        # whether normalize or not the output
        self.normalize = normalize
        # cluster centers to be used as codewords
        self.cluster_centers = cluster_centers

    def transform(self, diagrams):
        """
        Convert diagram or list of diagrams to their respective vectors.


        Parameters
        -----------
        diagrams : list of or singleton diagram, list of pairs. [(birth, death)]
            Persistence diagrams to be converted to persistence images. It is assumed they are
            in (birth, death) format. Can input a list of diagrams or a single diagram.

        """

        # if diagram is empty, return zero vector
        if len(diagrams) == 0:
            return np.zeros(self.N)

         # if first entry of first entry is not iterable, then diagrams is singular and we need
         # to make it a list of diagrams
        try:
            singular = not isinstance(diagrams[0][0], collections.Iterable)
        except IndexError:
            singular = False

        if singular:
            diagrams = [diagrams]

        dgs = [np.copy(diagram, np.float64) for diagram in diagrams]
        landscapes = [to_landscape(dg) for dg in dgs]

        # calculate cluster centers and return specific weightings
        weighting = self._getclustercenters(landscapes)

        wpbows = [self._transform(dgm, weighting[counter]) for counter, dgm in enumerate(landscapes)]

        # Make sure we return one item.
        if singular:
            wpbows = wpbows[0]

        return wpbows

    def _transform(self, landscape, weighting):
        """
        Calculate the weighted persistence bag of words vector for the specified landscape
        """
        wpbow_landscape = np.zeros(self.N)
        for counter, point in enumerate(landscape):
            wpbow_landscape[_distanceclustercenter(self, point)] += weighting[counter]

        if self.normalize:
            return wpbow_landscape/np.linalg.norm(wpbow_landscape)
        else:
            return wpbow_landscape

    def _getclustercenters(self, landscapes):
        """
        Cluster the consolidated diagram and return the cluster centers
        """
        # consolidate the landscapes
        consolidated_landscapes = _consolidate(self, landscapes)
        # get the 5th and 95th percentiles w.r.t. persistence points
        self.a, self.b = np.percentile(consolidated_landscapes[:,1], [5,95])
        # calculate the weight for every point with respect to the persistence.
        if not isinstance(self.n_subsample, int):
            weighting = [[1] * landscape.shape[0] for landscape in landscapes]
        else:
            weighting = [[weight_function(self, x[1]) for x in landscape] for landscape
                          in landscapes]
        if not isinstance(self.cluster_centers, list):
            # consolidate weighting
            consolidated_weighting = np.concatenate(weighting)
            # normalize weighting
            consolidated_weighting = consolidated_weighting/np.sum(consolidated_weighting)
            # subsample the points respecting the weighting
            if not isinstance(self.n_subsample, int):
                subsampled_points = range(consolidated_weighting.shape[0])
            else:
                subsampled_points = np.random.choice(consolidated_landscapes.shape[0],
                       size=self.n_subsample, replace=False, p=consolidated_weighting)
            # cluster using kmeans
            kmeans = KMeans(n_clusters = self.N).fit(consolidated_landscapes[subsampled_points])
            self.cluster_centers = kmeans.cluster_centers_
        return weighting
