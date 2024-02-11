import matplotlib.pyplot as plt
import numpy as np
import random
import math
from sklearn.mixture import GaussianMixture

def main():
    np.random.seed(123)

    # Prepare datasets
    dataset1 = sample_from_parent_distribution()
    dataset2 = perturb_dataset(dataset1)
    # dataset2 = estimate_and_sample(dataset1)
    
    # Calculate global bounds from all three datasets so they can be displaced with same axes limits
    tmp_dataset = np.vstack([dataset1, dataset2])  
    v1_min, v1_max = np.min(tmp_dataset[:,0]), np.max(tmp_dataset[:,0])
    v2_min, v2_max = np.min(tmp_dataset[:,1]), np.max(tmp_dataset[:,1])
    v3_min, v3_max = np.min(tmp_dataset[:,2]), np.max(tmp_dataset[:,2])
    bounds = [v1_min, v1_max, v2_min, v2_max, v3_min, v3_max]
    n_dims = tmp_dataset.shape[1]
    ranges = np.array([(np.max(tmp_dataset[:,x])-np.min(tmp_dataset[:,x])) if np.isreal(tmp_dataset[0,x]) else None for x in range(n_dims)])

    # Call the plot function
    plot_scatter(dataset1, dataset2, bounds)
    
    # Perform maximum similarity tests
    max_intra_sims_d1 = AverageMaxSimInternal(dataset1, ranges)
    max_intra_sims_d2 = AverageMaxSimInternal(dataset2, ranges)
    max_sims_cross_1 = AverageMaxSimCross(dataset2, dataset1, ranges)
    max_sims_cross_2 = AverageMaxSimCross(dataset1, dataset2, ranges)
    plot_hist(max_intra_sims_d1, max_intra_sims_d2, max_sims_cross_1, n_bins=30, vmin=0.8, vmax=1.0)

    print('Av max internal similarity Dataset 1', np.average(max_intra_sims_d1))
    print('Av max internal similarity Dataset 2', np.average(max_intra_sims_d2))
    print('Av max cross similarity Dataset 2 and Dataset 1', np.average(max_sims_cross_1))
    print('Av max cross similarity Dataset 1 and Dataset 2', np.average(max_sims_cross_2))

def plot_scatter(dataset1, dataset2, bounds):
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    plt.rcParams['font.size'] = 16

    # Set axis limits
    [v1_min, v1_max, v2_min, v2_max, v3_min, v3_max] = bounds
    # Plot points
    X1_1 = dataset1[np.where(dataset1[:,3]=='red')]
    X1_2 = dataset1[np.where(dataset1[:,3]=='blue')]
    X1_3 = dataset1[np.where(dataset1[:,3]=='green')]
    X2_1 = dataset2[np.where(dataset2[:,3]=='red')]
    X2_2 = dataset2[np.where(dataset2[:,3]=='blue')]
    X2_3 = dataset2[np.where(dataset2[:,3]=='green')]
    # select which variables to plot on which axes
    x = 1
    y = 0
    z = 2
    ax1.set_xlim((np.floor(v2_min), np.ceil(v2_max)))
    ax1.set_ylim((np.floor(v1_min), np.ceil(v1_max)))
    ax1.set_zlim((np.floor(v3_min), np.ceil(v3_max)))
    ax1.scatter(X1_1[:,x], X1_1[:,y], X1_1[:,z], s=2, color='red')
    ax1.scatter(X1_2[:,x], X1_2[:,y], X1_2[:,z], s=2, color='orange')
    ax1.scatter(X1_3[:,x], X1_3[:,y], X1_3[:,z], s=2, color='seagreen')
    ax1.tick_params(axis="x", labelsize=7)
    ax1.tick_params(axis="y", labelsize=7)
    ax1.tick_params(axis="z", labelsize=7)
    ax1.tick_params(axis='both', which='both', pad=-3)
    ax1.set_title("Dataset 1", fontsize = 11)

    ax2.set_xlim((np.floor(v2_min), np.ceil(v2_max)))
    ax2.set_ylim((np.floor(v1_min), np.ceil(v1_max)))
    ax2.set_zlim((np.floor(v3_min), np.ceil(v3_max)))
    ax2.scatter(X2_1[:,x], X2_1[:,y], X2_1[:,z], s=2, color='red')
    ax2.scatter(X2_2[:,x], X2_2[:,y], X2_2[:,z], s=2, color='orange')
    ax2.scatter(X2_3[:,x], X2_3[:,y], X2_3[:,z], s=2, color='seagreen')
    ax2.tick_params(axis="x", labelsize=7)
    ax2.tick_params(axis="y", labelsize=7)
    ax2.tick_params(axis="z", labelsize=7)
    ax2.tick_params(axis='both', which='both', pad=-3)
    ax2.set_title("Dataset 3", fontsize = 11)
    plt.tight_layout()
    plt.show()
    fig.savefig('scatterplots.png', dpi=300, bbox_inches="tight")

    
def plot_hist(max_sims_real, max_sims_synth, max_sims_cross, n_bins, vmin, vmax):    
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    plt.rcParams['font.size'] = 12
    n, bins, patches = ax1.hist(max_sims_real, n_bins, range=[vmin,vmax], density=True)
    tmp1 = np.max(n)
    n, bins, patches = ax2.hist(max_sims_synth, n_bins, range=[vmin,vmax], density=True)
    tmp2 = np.max(n)
    n, bins, patches = ax3.hist(max_sims_cross, n_bins, range=[vmin,vmax], density=True)
    tmp3 = np.max(n)
    y_lim = np.max([math.ceil(tmp1), math.ceil(tmp2), math.ceil(tmp3)])

    # Use these for the FIRST set of results
    ax1.set_xlabel('Max intra-set sims (Dataset 1)', fontsize=12)
    ax2.set_xlabel('Max intra-set sims (Dataset 3)', fontsize=12)
    ax3.set_xlabel('Max cross-set sims', fontsize=12)
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelsize=12)
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax3.tick_params(axis="x", labelsize=12)
    ax3.tick_params(axis="y", labelsize=12)

    ax1.set_ylim(bottom=0, top=y_lim+2)
    ax2.set_ylim(bottom=0, top=y_lim+2)
    ax3.set_ylim(bottom=0, top=y_lim+2)
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    fig.savefig('histograms.png', dpi=300, bbox_inches="tight")
    plt.show()

def AverageMaxSimInternal(data, ranges):
    # Must modify so that is not calculating similarity with itself
    n_points, n_dims = data.shape
    maximum_sims = []
    max = 0.0
    for i in range(n_points):
        # print(i)
        rest = list(range(n_points))
        rest.remove(i)
        sims = []
        for j in rest:
            sims.append(GowerSimilarity(data[i,:], data[j,:], ranges))
        maximum_sims.append(np.max(sims))
    return(maximum_sims)

def AverageMaxSimCross(samples, data, ranges):
    n_samples, n_dims = samples.shape
    n_points, n_dims = data.shape
    maximum_sims = []
    for i in range(n_samples):
        # print(i)
        sims = []
        for j in range(n_points):
            sims.append(GowerSimilarity(samples[i,:], data[j,:], ranges))
        maximum_sims.append(np.max(sims))
    return(maximum_sims)

def GowerSimilarity(pt1, pt2, ranges):
    # Returns Gower similarity between two points
    # Parameters:
    #   pt1 (1d numpy array): First datapoint 
    #   pt2 (1d numpy array): Second datapoint 
    #   ranges (1d numpy array): The ranges for numeric variables
    # Returns:
    #   sim (float): Gower similarity
    n_dims = pt1.size
    psi_sum = 0.0
    for i in range(n_dims):
        if ranges[i] != None: 
        # if np.isreal(pt1[i]):      
            psi = np.array(1.0-np.abs(pt1[i]-pt2[i])/ranges[i])
        else:
            psi = 1.0 if pt1[i]==pt2[i] else 0.0
        psi_sum = psi_sum + psi
    sim = psi_sum/n_dims
    return sim
    
def perturb_dataset(dataset):
    mean = [0, 0, 0]
    perturbed_dataset = dataset.copy()
    n_points = perturbed_dataset.shape[0]
    for i in range(n_points):
        # var = random.uniform(0.05, 0.05)
        var = 0.02
        cov = [[var, 0, 0], [0, var, 0], [0, 0, var]]  
        perturbation = np.random.multivariate_normal(mean, cov, 1)[0]
        # print(perturbation)
        # input('press')
        radius = np.sqrt(np.square(perturbation[0])+np.square(perturbation[1])+np.square(perturbation[2]))
        while radius < 0.05:
            perturbation = np.random.multivariate_normal(mean, cov, 1)[0]
            radius = np.sqrt(np.square(perturbation[0])+np.square(perturbation[1])+np.square(perturbation[2]))
        perturbed_dataset[i,0:3] = dataset[i,0:3] + perturbation
    return perturbed_dataset


def sample_from_parent_distribution():
    m1 = np.array([5.0, 3.5, 1.4])
    m2 = np.array([5.9, 2.8,  4.3 ])
    m3 = np.array([6.6, 3.0, 5.6])
    c1 = np.array([[0.122, 0.097, 0.016], [0.097, 0.141, 0.011], [0.016, 0.011, 0.030]])
    c2 = np.array([[0.261, 0.083, 0.179], [0.083, 0.096, 0.081], [0.179, 0.081, 0.216]])
    c3 = np.array([[0.396, 0.092, 0.297], [0.092, 0.102, 0.070], [0.297, 0.070, 0.298]])
    n1_samples = 50
    n2_samples = 50
    n3_samples = 50
    X1 = np.random.multivariate_normal(m1, c1, n1_samples)
    X2 = np.random.multivariate_normal(m2, c2, n2_samples)
    X3 = np.random.multivariate_normal(m3, c3, n3_samples)
    y1 = np.array(['red' for _ in range(n1_samples)],dtype=object).reshape((n1_samples,1))
    y2 = np.array(['blue' for _ in range(n2_samples)],dtype=object).reshape((n2_samples,1))
    y3 = np.array(['green' for _ in range(n3_samples)],dtype=object).reshape((n3_samples,1))
    X_all = np.vstack([X1,X2,X3])
    y_all = np.vstack([y1,y2,y3])
    dataset = np.concatenate((X_all, y_all), axis=1)
    return dataset    

def estimate_and_sample(dataset):
    X = dataset[:, 0:3]
    y = dataset[:, 3]
    gm1 = GaussianMixture(n_components=1).fit(X[0:50,0:3])
    gm2 = GaussianMixture(n_components=1).fit(X[50:100,0:3])
    gm3 = GaussianMixture(n_components=1).fit(X[100:150,0:3])
    n1_samples = 50
    n2_samples = 50
    n3_samples = 50
    X1 = gm1.sample(50)[0]
    X2 = gm2.sample(50)[0]
    X3 = gm3.sample(50)[0]
    y1 = np.array(['red' for _ in range(n1_samples)],dtype=object).reshape((n1_samples,1))
    y2 = np.array(['blue' for _ in range(n2_samples)],dtype=object).reshape((n2_samples,1))
    y3 = np.array(['green' for _ in range(n3_samples)],dtype=object).reshape((n3_samples,1))
    X_all = np.vstack([X1,X2,X3])
    y_all = np.vstack([y1,y2,y3])
    new_dataset = np.concatenate((X_all, y_all), axis=1)
    return new_dataset

    
if __name__ == "__main__":
    main()
    
    
    
