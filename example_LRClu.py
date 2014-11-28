"""
LRClu
"""
import numpy
import scipy
import nibabel
from nilearn import datasets

def hardthresh(mat):
    """
    keeps only the greatest coefficient of each column of mat    
    """
    _,nbcol=mat.shape
    for i in range(0,nbcol):
        mat[:,i]=(mat[:,i]==mat[:,i].max()).astype(int)*mat[:,i]
    return mat
 

def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
    Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
    Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
    Dice coefficient as a float on range [0,1].
    Maximum similarity = 1
    No similarity = 0
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
     
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
     
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
     
    return 2. * intersection.sum() / (im1.sum() + im2.sum()) 

class LRClu():
    """performs low rank clustering
    """
    def __init__(self):
        self.K=10
        self.maxit=5
        self.X4d=nibabel.load(datasets.fetch_adhd().func[0]).get_data()
        x,y,z,N = X4d.shape
        M=x*y*z
        self.X=X4d.reshape((M,N))
        
        #random initialisation
        self.D=numpy.random.rand(M,K)
        self.S=numpy.random.rand(K,N)
        self.S=hardthresh(self.S)
        self.S0=self.S
    def factorize(self):
        
        for it in range(0,self.maxit):
            
            print it
            print dice(self.S,self.S0)
            #evasive action : drop empty clusters
            nnz=abs(self.S.transpose()).sum(axis=0).nonzero()[0]
            iS=self.S[nnz,:]
            iD=self.D[:,nnz]
            #update the centers
            self.D[:,nnz]=iD+numpy.dot(self.X-numpy.dot(iD,iS),iS.transpose())
            
            #clear iS
            #normalize D
            for k in range(0,K):
                self.D[:,k]=self.D[:,k]/numpy.linalg.norm(self.D[:,k])
                
            #update the assignments
            self.S=self.S+numpy.dot(self.D.transpose(),self.X-numpy.dot(self.D,self.S));
            self.S=hardthresh(self.S)
            
DS=LRClu()
DS.factorize()



    
    


    


# Author: Alexis Benichoux