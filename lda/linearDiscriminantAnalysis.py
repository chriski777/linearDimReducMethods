import numpy as np
from scipy.linalg import lstsq

class Model:
    
    def __init__(self, n_components):
        if n_components != 1:
            raise ValueError("Multiple Discriminant Analysis not implemented")
        self.n_components = n_components
        self.fitted = False
        
    def fit(self, trainX, trainY):
        """
        Fits a linear discriminant model to provided trainX and trainY.
        Related to the sklearn.LinearDiscriminantAnalysis using an eigen solver.
        
        Based on section 4.11 of 
        References
        ----------
        .. [1] R. O. Duda, P. E. Hart, D. G. Stork. Pattern Classification
           (Second Edition). John Wiley & Sons, Inc., New York, 2001. ISBN
           0-471-05669-3.

        Parameters
        ---------
        trainX:
            Array of training data (n_samples, n_features)
        trainY:
            Array of labels for training data (n_samples, )
        """
        # Identify total number of classes
        self.uniqueClassLabels = np.unique(trainY)
        numUniqueClassLabels = len(self.uniqueClassLabels)
        self.numPerClass = np.zeros((numUniqueClassLabels, ))
        num_features = trainX.shape[1]
        num_samples = trainX.shape[0]
        # Keep track of class means and covariances
        self.classMeans = np.full([numUniqueClassLabels, num_features], np.nan)
        self.classScatterMats = np.full([numUniqueClassLabels, num_features, num_features], np.nan)
        # Compute means and covariance matrices for each class
        for i in range(numUniqueClassLabels):
            currClassLabel = self.uniqueClassLabels[i]
            # Find all indices of training data that matches current class
            classIdx = np.where(trainY == currClassLabel)[0]
            currClassSamples = trainX[classIdx,:]
            # Keep track of how many samples are in class during training
            self.numPerClass[i] = len(classIdx)
            # Identify class means (n_labels, n_features)
            self.classMeans[i,:] = np.mean(currClassSamples, axis=0)
            # Identify class scatter matrices (n_features, n_features) 
            # By setting bias = true, we normalize by N instead of N-1 (Bessel's correction). 
            # rowvar = false means each row is a sample
            self.classScatterMats[i, :, :] = self.numPerClass[i]*np.cov(currClassSamples, rowvar=False, bias=True)
        self.totalMeanVector = np.mean(trainX, axis=0)
        # Generate within-class scatter matrix which is the sum of all class' scatter matrices
        self.within_class_scatter = np.sum(self.classScatterMats, axis=0)
        # Generate between-class scatter matrix
        # Within-class scatter matrix and between-class scatter matrix differ from these by a factor of num_samples
        # eigenvalues found are identical to sklearn.lda eigensolver implementation though 
        self.between_class_scatter = np.zeros((num_features, num_features))
        for i in range(numUniqueClassLabels):
            v = (self.classMeans[i,:] - self.totalMeanVector)[:,np.newaxis]
            self.between_class_scatter += self.numPerClass[i]*(v @ v.T)
        #find projection vector and project trainX
        # If Sw is singular, use the least squres estimate of the inverse instead of throwing an error 
        invSwSb, _, _ , _= lstsq(self.within_class_scatter, self.between_class_scatter)
        U, _, _ = np.linalg.svd(invSwSb)
        self.scalings_ = U
        self.fitted = True

    def transform(self, X):
        """
        Projects data to lower dimensional subspaces found during model fitting
        
        Parameters
        ---------
        X:
            Array of data you'd like to project on to dimensions found during fitting (n_samples, n_features) 
        
        Returns 
        ---------
        proj: 
            Data projections 
        """
        if self.fitted:
            # Loadings are in ascending order
            return X@(self.scalings_[:,0:self.n_components])
        else:
            raise Exception('Model has not been fitted yet!')
