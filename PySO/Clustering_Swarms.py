import numpy as np

import matplotlib.pyplot as plt

import warnings as warning

from kneed import KneeLocator

from scipy.signal import argrelextrema
from scipy.interpolate import interp1d

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans


def Standardize(X):
    """
    Standardize features - subtract mean and scale to unit variance

    INPUTS
    ------
    X: array, shape=(n,p)
        the positions of the n particles in the p-dimensional space

    RETURNS
    -------
    X_std: array, shape=(n,p)
        the positions of the n particles in the p-dimensional space
        standardised to zero-mean and unit-variance form
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler.transform(X, copy=True)


def KMeans_clustering(X_std, K, use_minibatch=False, batch_size=1024):
    """
    Apply the KMeans clustering algorithm

    INPUTS
    ------
    X_std: array, shape=(n,p)
        the positions of the n particles in the p-dimensional space
        standardised to zero-mean and unit-variance form
    K: int
        the number of clusters to use
    use_minibatch: bool
        whether to use MiniBatchKMeans for faster clustering on large datasets
    batch_size: int
        batch size for MiniBatchKMeans (ignored if use_minibatch=False)

    RETURNS
    -------
    I: float
        the inertia of the clustering; i.e. the sum of the squared
        distances between the points and the cluster centres
    membership: array, shape=(n,)
        for each particle, return an int in the range 0, K-1
        identifying the cluster to which it belongs
    """
    if use_minibatch:
        kmeans = MiniBatchKMeans(n_clusters=K, n_init=3, batch_size=batch_size, random_state=42)
    else:
        kmeans = KMeans(n_clusters=K, n_init='auto', random_state=42)
    
    kmeans.fit(X_std)

    I = kmeans.inertia_
    memberships = kmeans.labels_

    return I, memberships, kmeans


def NumberOfClusters(X_std,
                     min_clusters, max_clusters,
                     save_elbow_curve=None,
                     elbow_tol=1.0):
    """
    Find the optimum number of clusters in the data X_std

    This is an attempt to automate the "elbow method", but there
    is really no substitute for looking the plot using save_elbow_curve

    INPUTS
    ------
    X_std: array, shape=(n,p)
        the positions of the n particles in the p-dimensional space
        standardised to zero-mean and unit-variance form
    min_clusters, max_clusters: int
        if K is not provided then these instead give
        the range of the number of clusters to try
    save_elbow_curve: str
        if None, then do not produce the elbow curve plot, else
        this argument is the filname where plot is to be saved
    elbow_tol: float
        sensitivity parameter for elbow detection. "Put simply,
        S is a measure of how many “flat” points we expect to see
        in the unmodified data curve before declaring a knee"
        see https://raghavan.usc.edu//papers/kneedle-simplex11.pdf

    RETURNS
    -------
    K: int
        the number of clusters to use
    """
    # Optimize: test fewer K values using log-scale sampling for large ranges
    # Use MiniBatchKMeans for datasets with > 10000 points
    k_range = max_clusters - min_clusters + 1
    max_k_trials = 20  # Maximum number of K values to test
    use_minibatch = (X_std.shape[0] > 10000)
    
    if k_range > max_k_trials:
        # Sample K values on log scale for efficiency
        nclusters = np.unique(np.logspace(
            np.log10(min_clusters),
            np.log10(max_clusters),
            num=min(max_k_trials, k_range)
        ).astype(int))
    else:
        nclusters = np.arange(min_clusters, max_clusters+1)
    
    # apply KMeans clustering for trial number of clusters and record inertia
    distorsions = np.zeros(len(nclusters), dtype=float)
    for i, k in enumerate(nclusters):
        I, _, _ = KMeans_clustering(X_std, k, use_minibatch=use_minibatch)
        distorsions[i] = np.sqrt(I)

    # find the 'elbow' in the inertia curve
    knee_loc = KneeLocator(nclusters, distorsions, S=elbow_tol,
                           curve="convex", direction="decreasing")
    if knee_loc.knee:
        K = knee_loc.knee
    else:
        W = "Can't determine Nclusters, using min value K="+str(min_clusters)
        warning.warn(W)
        K = min_clusters

    # make summary 'elbow' plot for visual inspection
    if save_elbow_curve is not None: #
        plt.plot(nclusters, distorsions, 'b-+')
        plt.axvline(K, c='k', ls=':', label='K')
        plt.xticks(nclusters)
        plt.xlabel('Number of clusters')
        plt.ylabel('Sqrt(Inertia)')
        plt.title('Elbow Curve')
        plt.legend()
        plt.savefig(save_elbow_curve)
        plt.clf()

    return K


def RemoveSmallClusters(X_std, kmeans, min_membership):
    """
    Any clusters below a certain size are removed and their members
    assigned to the nearest cluster
    
    INPUTS
    ------
    X_std: array, shape=(n,p)
        the positions of the n particles in the p-dimensional space
        standardised to zero-mean and unit-variance form
    kmeans: K-Means
        instance of sklearn K-Means clustering class
    min_membership: int
        the minimum number of points in a cluster

    RETURNS
    -------
    K: int
        the number of clusters used by KMeans
    memberships: array, shape=(n,)
        for each particle, return an int identifying the cluster
        to which it belongs
    """
    unique, unique_counts = np.unique(kmeans.labels_, return_counts=True)

    # Vectorized check for small clusters
    small_clusters_mask = (unique_counts < min_membership)
    
    if not np.any(small_clusters_mask):
        # all the clusters are already of an appropriate size... do nothing
        return kmeans.n_clusters, kmeans.labels_

    if np.all(small_clusters_mask):
        # all of the clusters are too small... oh dear
        W = "Clustering failed:"
        W += " all clusters below min size = {}".format(min_membership)
        W += " (returning points with no clustering)"
        warning.warn(W)
        return 1, np.zeros(len(kmeans.labels_), dtype=int)

    # keep only cluster above min size
    big_clusters = ~small_clusters_mask
    kmeans.cluster_centers_ = kmeans.cluster_centers_[big_clusters]

    W = "Removing {0} clusters with {1} particles ".format(
                        np.sum(small_clusters_mask),
                        unique_counts[small_clusters_mask])
    warning.warn(W)

    K = np.sum(big_clusters)
    memberships = kmeans.predict(X_std)

    return K, memberships


def Clustering(X,
               K=None,
               min_clusters=1, max_clusters=40,
               min_membership=None,
               save_elbow_curve=None,
               elbow_tol=1.0):
    """
    Function for using KMeans clustering to split up our swarms

    The number of clusters can be detected automatically using the
    "elbow method" by specifying a search range (min/max_clusters)
    and a tolerance parameter (elbow_tol)

    INPUTS
    ------
    X: array, shape=(n,p)
        the positions of the n particles in the p-dimensional space
    K: int
        the number of clusters to use, if known in advance
    min_clusters, max_clusters: int
        if K is not provided then these instead give
        the range of the number of clusters to try
    min_membership: int or None
        the minimum number of points in a cluster
        for example, if this is set to 5, then any clusters with 4 or
        fewer members will be disbanded and the members reassigned to the
        nearest remaining cluster
    save_elbow_curve: str
        if None, then do not produce the elbow curve plot, else
        this argument is the filname where plot is to be saved
    elbow_tol: float
        sensitivity parameter for elbow detection. "Put simply,
        S is a measure of how many “flat” points we expect to see
        in the unmodified data curve before declaring a knee"
        see https://raghavan.usc.edu//papers/kneedle-simplex11.pdf

    RETURNS
    -------
    num_clusters: int
        the number of clusters used by KMeans
    cluster_membership: array, shape=(n,)
        for each particle an int identifies the cluster to which it belongs
    kmeans: K-Means
        instance of sklearn K-Means clustering class
    """
    # check input has the correct shape
    assert X.ndim==2, "X.shape={} is not of form (n,p)".format(X.shape)
    
    n_samples = X.shape[0]
    
    # Early return for trivial cases
    if min_membership and n_samples < min_membership * 2:
        return 1, np.zeros(n_samples, dtype=int)

    # standardize the data to zero mean and unit variance
    X_std = Standardize(X)
    
    # Automatically use MiniBatchKMeans for large datasets (>10k points)
    use_minibatch = (n_samples > 10000)

    if K is None:
        # Limit max_clusters to reasonable value based on min_membership
        if min_membership:
            max_clusters = min(max_clusters, n_samples // min_membership)
        
        # determine the number of clusters to use
        K = NumberOfClusters(X_std, min_clusters, max_clusters,
                             save_elbow_curve=save_elbow_curve,
                             elbow_tol=elbow_tol)

    # run KMeans clustering on the data
    _, memberships, kmeans = KMeans_clustering(X_std, K, use_minibatch=use_minibatch)

    # enforce minimum cluster membership size
    if min_membership:
        K, memberships = RemoveSmallClusters(X_std, kmeans, int(min_membership))

    return K, memberships

def main():
    """
    THIS IS A UNIT TEST
    """
    ndim = 5
    nclusters = 4
    npoints = 1000

    component_dists = []
    for i in range(nclusters):
        mean = np.random.uniform(-12, 12, size=ndim)
        cov = np.eye(ndim)
        component_dists.append(MultivariateGaussianDistribution(mean, cov))

    mixture_model = GeneralMixtureModel(component_dists,
                                        weights=np.ones(nclusters))

    pts = mixture_model.sample(npoints)

    pts = np.vstack((pts, 30*np.ones(ndim)))

    # Apply our clustering algorithm, searching for between 1 and 100 clusters
    # (take a look at the intertia plot saved in file test_elbow.png)
    K, memberships = Clustering(pts, min_clusters=1, max_clusters=20,
                    min_membership=10, save_elbow_curve='test_elbow.png')

    # Plot the data, color coded by the detected cluster membership
    # (take a look at the intertia plot saved in file test_clusters.png)
    fig1, axes = plt.subplots(nrows=ndim, ncols=ndim, figsize=(10, 10))
    for col in np.arange(ndim):
        for row in np.arange(ndim):

            for k in range(K):
                mask = (memberships==k)
                axes[ndim-row-1,col].scatter(pts[mask][:,col], pts[mask][:,row],
                                              marker='.', c='C'+str(k))

            axes[ndim-row-1, col].set_xlabel("x"+str(col))
            axes[ndim-row-1, col].set_ylabel("x"+str(row))

            axes[ndim-row-1, col].set_xlim(-31, 31)
            axes[ndim-row-1, col].set_ylim(-31, 31)

    plt.tight_layout()
    plt.savefig("test_clusters.png")
    plt.clf()

    # Ideally, we should find the same number of clusters as there
    # were components in out original Gaussian mixture model
    print("clusters detected =", K, "of", nclusters)
    print("cluster sizes =", np.unique(memberships, return_counts=True)[1])

if __name__=='__main__':

    main()
