import numpy
import scipy
import nibabel
from nilearn import datasets, input_data
import pickle


def hardthresh(mat):
    """
    hard thresholding, corresponding to 1-sparse contraint for the
    assignment matrix
    keeps only the greatest coefficient of each column of mat
    """
    _, nbcol = mat.shape
    for i in range(0, nbcol):
        mat[:, i] = (mat[:, i] == mat[:, i].max()).astype(int)*mat[:, i]
    return mat


class LRClu():
    """
    Spatially constrained clustering, using a sparse + low rank tv regularised
    factorization

    Benichoux, A. and Blumensath, T. (2014)
    A spatially constrained low-rank matrix factorization for the
    functional parcellation of the brain
    Proc. 22nd European Signal Processing Conference, pp. 1-5.
    """

    def __init__(self, dataset):

        self.K = 100
        self.maxit = 100
        # written to fit nilearn HCP dataset format, any numpy array should do
        # self.dataset = datasets.fetch_adhd()
        # X4d = nibabel.load(datasets.fetch_adhd().func[0]).get_data()
        self.dataset = dataset
        x, y, z, M = X4d.shape
        N = x*y*z
        self.brainsize = [x, y, z]
        self.X = X4d.reshape((N, M)).transpose()
        self.tv = 0
        self.eps = 0.02
        # random initialisation
        self.D = numpy.random.rand(M, self.K)
        self.S = numpy.random.rand(self.K, N)
        self.S = hardthresh(self.S)
        self.S0 = self.S  # to be replaced by ground truth, if available
        if self.tv:
            try:
                self.nbh = pickle.load(open('neigh' + str(N)+'.pickle', 'rb'))

                print "Neighbourhood table found"
            except:
                self.getNeighbours()

    def getNeighbours(self):
        """
        computes the neighbourhood table, corresponding to the spatial prior
        """
        N = numpy.prod(self.brainsize)
        E = [-1, 0, 1]
        self.nbh = numpy.zeros((N, 26))
        for k in range(0, N):
            if numpy.mod(k, int(N/10)) == 0:
                print 'Building neighbourhood table : ' + str(100*k/N) + '%'

            xk, yk, zk = numpy.unravel_index(k, self.brainsize)
            neigh_id = 0
            for ex in E:
                for ey in E:
                    for ez in E:
                        if abs(ex)+abs(ey)+abs(ez):
                            nesub = [max(min(xk+ex, self.brainsize[0]-1), 0), max(
                                min(yk+ey, self.brainsize[1]-1), 0), max(min(zk+ez, self.brainsize[2]-1), 0)]
                            neigh = numpy.ravel_multi_index(
                                nesub, self.brainsize)
                            self.nbh[k, neigh_id] = neigh
                            neigh_id = neigh_id+1
        pickle.dump(Nbh, open('neigh' + str(N)+'.pickle', 'wb'))

    def tv_grad(self, S1d):
        """
        computes gradient of smoothed tv
        """
        N = len(S1d)
        Gr = numpy.zeros((N, 26))
        Y = numpy.zeros((N, 1))
        for n in range(0, N):
            nbhs = self.nbh[n, :]
            for k in range(0, len(nbhs)):
                Gr[n, k] = S1d[n]-S1d[nbhs[k]]
        d2 = (Gr*Gr).sum(axis=1)

        for n in range(0, N):

            nbhs = self.nbh[n, self.nbh[n, :].nonzero()][0]

            for k in range(0, len(nbhs)):

                Y[nbhs[k]] = Y[nbhs[k]]+S1d[nbhs[k]] / \
                    numpy.sqrt(self.eps**2+d2[n])

        G = -(26*S1d/numpy.sqrt(self.eps**2+d2)+Y).transpose()
        return G

    def factorize(self):
        """
        performs low rank clustering
        """
        for it in range(0, self.maxit):
            if numpy.mod(it, self.maxit/10) == 0:
                print 'Factorization : ' + str(100*it/self.maxit) + '%'
            # evasive action : drop empty clusters
            nnz = abs(self.S.transpose()).sum(axis=0).nonzero()[0]
            iS = self.S[nnz, :]
            iD = self.D[:, nnz]

            # update the centers
            self.D[:, nnz] = iD + \
                numpy.dot(self.X-numpy.dot(iD, iS), iS.transpose())

            # could clear iS

            # normalize D
            for k in range(0, self.K):
                self.D[:, k] = self.D[:, k]/numpy.linalg.norm(self.D[:, k])

            # update the assignments
            self.S = self.S + \
                numpy.dot(self.D.transpose(), self.X-numpy.dot(self.D, self.S))
            self.S = hardthresh(self.S)
            if self.tv:
                for k in range(0, self.K):

                    self.S[k, :] = self.S[k, :] - \
                        self.tv*self.tv_grad(self.S[k, :])

    def getBrainMap(self):
        K, N = self.S.shape
        self.brainmap = numpy.zeros(self.brainsize)
        for n in range(0, N):
            x, y, z = numpy.unravel_index(n, self.brainsize)
            self.brainmap[x, y, z] = sum(self.S[:, n].nonzero()[0])

 # Display clustering


from nilearn.image import mean_img
from nilearn.plotting.img_plotting import plot_roi, plot_epi


DS = LRClu()
DS.factorize()
DS.getBrainMap()

for k in range(1, 10):
    x0 = k*DS.brainsize[0]/10
    slice = DS.brainmap[x0, :, :]
    plt.imshow(slice)
    fig = plt.figure()

