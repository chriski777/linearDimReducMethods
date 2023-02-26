import numpy as np
import matplotlib.pyplot as plt
from lda import LinearDiscriminantAnalysis

def runExample():
    nDim = 2
    mean1 = (1,1)
    cov1 = [[1, 0], [0, 1]]
    mean2 = (15,1)
    cov2 = [[1, 0], [0, 1]]
    numSamples1 = 1000
    numSamples2 = 750
    sample1 = np.random.multivariate_normal(mean1, cov1, numSamples1)
    sample2 = np.random.multivariate_normal(mean2, cov2, numSamples2)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.title('Data')
    plt.scatter(sample1[:,0], sample1[:,1])
    plt.scatter(sample2[:,0], sample2[:,1])

    allSamples = np.vstack([sample1, sample2])
    allLabels = np.hstack([np.repeat(1, sample1.shape[0]), np.repeat(2, sample2.shape[0])])
    from sklearn.utils import shuffle
    allSamples, allLabels = shuffle(allSamples, allLabels, random_state=0)

    model = LinearDiscriminantAnalysis(n_components=1)
    model.fit(allSamples, allLabels)
    plt.figure()
    plt.title('Projection on to first LDA component')
    plt.scatter(model.transform(sample1), np.zeros((len(sample1),)))
    plt.scatter(model.transform(sample2), np.zeros((len(sample2),)))
    plt.show()
    
if __name__ == "__main__":
    runExample()