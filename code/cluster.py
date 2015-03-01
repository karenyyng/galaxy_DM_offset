"""cluster object that holds all the final info for comparison among
different clusters
"""
from tables import Float32Col, Int32Col, IsDescription


# maybe best organized as a dict first ...?
# then use pd.concat
class cluster(IsDescription):
    """processed info about a cluster
    * good for parallelization
    -[ ] think about projections
    """
    clstNo = Int32Col()
    relaxedness1 = Float32Col()
    relaxedness2 = Float32Col()
    galDomPeaks = Int32Col()
    galpeaks = Int32Col()
    DMpeaks = Int32Col()  # nested
    galPeakNo = Int32Col()
    DMPeakNo = Int32Col()
    richness = Float32Col()
