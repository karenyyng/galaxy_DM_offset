"""cluster object that holds all the final info for comparison among
different clusters
"""

class cluster():
    """processed info about a cluster
    this object
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
