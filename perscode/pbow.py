from __future__ import division
import collections

import numpy as np

from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans

from .utils import to_landscape, _consolidate, _distanceclustercenter

__all__ = ['PBoW']

class PBoW(TransformerMixin):
    """ Initialize a Persistence Bag of Words generator.


    Parameters
    -----------
    N : Length of codebook (length of vector)
    normalize : if set to True normalizes each vectorized persistence diagram with respect
                to the euclidean norm
    cluster_centers : list or numpy array of 2d vectors representing the cluster centers. The
                      length of the list should be equal to N.


    Usage
    -----------
    >>> import perscode
    >>> # define length of codebook
    >>> length_codebook = 10
    >>> pbow = perscode.PBoW(N = length_codebook)
    >>> pbow_diagrams = pbow.transform(diagrams) # diagrams is a list of persistence diagrams

    """
    def __init__(
        self,
        N = 30,
        normalize = False,
        cluster_centers = None,
        ):
        # size of codebook
        self.N = N
        # whether normalize or not the pbow vector
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

        # calculate cluster centers
        if self.cluster_centers == None:
            self._getclustercenters(landscapes)
        else:
            # check if cluster centers is of same size of codewords if cluster_center != None
            if len(self.cluster_centers) != self.N:
                raise Exception('The number of cluster centers is not compatible with N, the\
                                number of codewords')

        pbows = [self._transform(dgm) for dgm in landscapes]

        # Make sure we return one item.
        if singular:
            pbows = pbows[0]

        return pbows

    def _transform(self, landscape):
        """
        Calculate the persistence bag of words vector for the specified landscape
        """
        pbow_landscape = np.zeros(self.N)

        for point in landscape:
            pbow_landscape[_distanceclustercenter(self, point)] += 1

        if self.normalize:
            return pbow_landscape/np.linalg.norm(pbow_landscape)
        else:
            return pbow_landscape

    def _getclustercenters(self, landscapes):
        """
        Cluster the consolidated diagram and return the cluster centers
        """
        consolidated_landscapes = _consolidate(self, landscapes)
        kmeans = KMeans(n_clusters = self.N, n_jobs=-1).fit(consolidated_landscapes)
        self.cluster_centers = kmeans.cluster_centers_
