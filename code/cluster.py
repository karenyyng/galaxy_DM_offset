"""cluster object that holds all the final info for comparison among
different clusters

This class takes inspiration from https://github.com/pydata/pandas/issues/3032
"""
from tables import Float32Col, Int32Col, IsDescription


# maybe best organized as a dict first ...?
# then use pd.concat
class cluster():  # IsDescription
    """This stores all the irregular pieces of the fhat data
    """
    # clstNo = Int32Col()
    # relaxedness1 = Float32Col()
    # relaxedness2 = Float32Col()
    # galDomPeaks = Int32Col()
    # galpeaks = Int32Col()
    # DMpeaks = Int32Col()  # nested
    # galPeakNo = Int32Col()
    # DMPeakNo = Int32Col()  # just a number
    # richness = Float32Col()  # just a number

    def __init__(self, fhat):
        self.peaks_dens = fhat["peaks_dens"]
        self.peaks_coords0 = fhat["peaks_xcoords"]
        self.peaks_coords1 = fhat["peaks_ycoords"]
        self.xi = fhat["xi"]
        self.phi = fhat["phi"]
        pass
