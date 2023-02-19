import os
import numpy as np
from skimage.feature import hog
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


def unpickle(file):
    '''  Un-pack data '''
    import pickle
    with open(file, 'rb') as fo:
        ds = pickle.load(fo, encoding='bytes')
    return ds


def pca(train_data, test_data=[], min_pov=0.90):
    """ Apply PCA to dataset """
    def propose_suitable_d(eigenvalues):
        """ Propose a suitable d using POV = 95% """
        sum_D = sum(eigenvalues)
        for d in range(0, len(eigenvalues)):
            pov = sum(eigenvalues[:d])/sum_D
            if pov > min_pov:
                return d

    # Zero-mean train data and test data
    samples = mean_subtraction(train_data, test_data)
    samples = np.asarray(samples)
    cov = np.dot(samples.T, samples) / samples.shape[0]

    eigenvectors, eigenvalues, _ = np.linalg.svd(cov)

    samples = np.dot(samples, eigenvectors)  # decorrelate the data

    d = propose_suitable_d(eigenvalues)  # find suitable d

    # samples_pca becomes [N x d]
    samples_pca = np.dot(samples, eigenvectors[:, :d])
    return samples_pca, samples, eigenvalues


def pca_whitening(train_data, test_data=[], min_pov=0.90):
    """ whiten the data,
        The whitened data will be a gaussian with zero mean and identity covariance matrix.
    """
    _, samples, eigenvalues = pca(train_data, test_data, min_pov=min_pov)
    # whiten the data:
    # divide by the eigenvalues (which are square roots of the singular values)
    samples_pca_white = samples / np.sqrt(eigenvalues + 1e-5)
    return samples_pca_white


def mean_subtraction(train_data, test_data):
    """ Zero mean samples """
    train_data_mean = np.mean(train_data, axis=0)

    train_data -= train_data_mean
    if test_data:
        test_data -= train_data_mean
        samples = train_data.tolist()+test_data.tolist()
    else:
        samples = train_data
    return samples


def mean_normalization(train_data, test_data):
    """ Normalizes samples with Mean and Std"""
    samples = mean_subtraction(train_data, test_data)
    samples /= np.std(samples, axis=0)

    return samples


def lda(train_data, train_labels, test_data, test_labels):
    """ Apply LDA to dataset """
    def compute_mean_cov(dataset):
        """ Calculate covariance matrix for dataset matrix """
        covariance_matrix = None
        # Calculate covariance matrix
        covariance_matrix = np.cov(dataset, rowvar=False)

        # check Singularity for covariance matrix
        if np.linalg.det(covariance_matrix) == 0.0:
            row = covariance_matrix.shape[0]
            for i in range(row):
                covariance_matrix[i, i] += 0.0001

        mean_vector = np.mean(dataset, axis=0)

        return covariance_matrix, mean_vector

    train_images_dic, test_images_dic = {}, {}
    for i in range(10):
        train_images_dic[i] = []
        test_images_dic[i] = []

    for data, label in zip(train_data, train_labels):
        train_images_dic[label].append(data)
    for data, label in zip(test_data, test_labels):
        test_images_dic[label].append(data)

    s_w = None
    s_b = 0
    total_mean = None
    calsses_info = []

    for key, value in train_images_dic.items():
        calsses_info.append(compute_mean_cov(value))

    lst_cov, lst_mean = zip(*calsses_info)

    # convert tuple to list
    lst_cov = list(lst_cov)
    lst_mean = list(lst_mean)

    s_w = sum(lst_cov)

    total_mean = sum(lst_mean)/len(lst_mean)

    for i in range(len(calsses_info)):
        s_b += 2000*(lst_mean[i]-total_mean) * \
            (lst_mean[i]-total_mean).transpose()

    w = np.linalg.inv(s_w)*s_b

    # Compute eigenvector and eigenvalue of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(w)

    # Sort eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    w = eigenvectors[:9]
    samples_lda = np.matmul(np.asarray(
        train_data.tolist()+test_data.tolist()), w.T)
    # samples_lda = np.matmul(np.asarray(train_data+test_data), w.T)
    return samples_lda


def filter_feature_selection(dataset, n):
    dataset_filter, rslt = [], []
    dataset = np.asarray(dataset).T
    for i in range(len(dataset)):
        feature = dataset[i]
        rslt.append(np.var(feature))
    # best_features = sorted(rslt, reverse=True)[:n]
    rslt = np.array(rslt)
    # best_features_idx = rslt.argsort()[::-1]
    best_features_idx = rslt.argsort()[::-1]
    # print(rslt[best_features_idx[0]])
    for idx in best_features_idx[:n]:
        dataset_filter.append(dataset[idx])

    dataset_filter = np.asarray(dataset_filter)
    return dataset_filter.T


def simple_hog_features(dataset):
    rslt = []
    for img in dataset:
        img = np.asarray(img).reshape((32, 32, 3))
        _, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), visualize=True, multichannel=True)

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(
            hog_image, in_range=(0, 10))
        rslt.append(hog_image_rescaled)
    return rslt


def SIFT(dataset):
    # import cv2
    from siftdetector import detect_keypoints

    rslt = []

    for img in dataset:
        img = np.asarray(img).reshape((32, 32, 3))
        [_, descriptors] = detect_keypoints(img, 5)

        rslt.append(descriptors)
    return rslt
