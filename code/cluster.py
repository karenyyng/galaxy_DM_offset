"""cluster object that holds all the final info for comparison among
different clusters
"""


class cluster():
    """processed info about a cluster
    * good for parallelization
    -[ ] think about projections
    """
    def __init__(self, clstNo):
        self.clstNo = clstNo
        self.__relaxedness1__ = None
        self.__relaxedness2__ = None
        self.__galDomPeaks__ = None
        self.__galpeaks__ = None
        self.__DMpeaks__ = None
        self.__galPeakNo__ = len(self.__galpeaks__)
        self.__DMPeakNo__ = len(self.__DMpeaks__)
        self.__richness__ = None

    def compute_galDMoffsets(self):
        return

    def compute_relaxedness1(self):
        """amount of mass in subhalos / amount of mass of the cluster"""
        return

    def compute_relaxedness2(self):
        """distance between the CM and the most bound particle of the
        density profile in terms of the r200C"""

        return

