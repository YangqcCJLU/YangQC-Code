from scipy.io import loadmat
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from functools import reduce
import spectral
import cv2


# SG smooth
def SG_smooth(data_D):
    return savgol_filter(data_D, 5, 3, mode='nearest')


# Norm
def Normalization(data_D):
    return (data_D - np.min(data_D, axis=1, keepdims=True)) / (
            np.max(data_D, axis=1, keepdims=True) - np.min(data_D, axis=1, keepdims=True))


# Standardization
def Standardization(data_D):
    return preprocessing.StandardScaler().fit_transform(data_D)


# MSC
def MSC(data_D):
    mean = np.mean(data_D, axis=0)
    data_D = np.zeros_like(data_D)
    for i in range(data_D.shape[0]):
        p = np.polyfit(mean, data_D[i, :], 1)
        data_D[i] = (data_D[i] - p[1]) / p[0]
    return data_D


# SNV
def SNV(data_D):
    mean2 = np.mean(data_D, axis=1)
    std = np.std(data_D, axis=1)
    return (data_D - mean2[:, np.newaxis]) / std[:, np.newaxis]


# FD
def FD(data_D):
    return np.gradient(data_D, axis=1)


# Visualization of anomalous samples
def visualize_3D(data, labels):
    normal_data = data[labels == 1]
    anomaly_data = data[labels == -1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(normal_data[:, 0], normal_data[:, 1], normal_data[:, 2], c='blue', label='Normal')
    ax.scatter(anomaly_data[:, 0], anomaly_data[:, 1], anomaly_data[:, 2], c='red', label='Abnormal')
    ax.set_xlabel('PC1', fontsize=16)
    ax.set_ylabel('PC2', fontsize=16)
    ax.set_zlabel('PC3', fontsize=16)
    ax.grid(False)
    # Set the axes background color to white
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    legend = ax.legend()
    for text in legend.get_texts():
        text.set_fontsize(16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    ax.set_facecolor('white')
    plt.savefig("IF.tif", dpi=300)
    plt.tight_layout()
    plt.show()


def visualize_2D(data, labels):
    normal_data = data[labels == 1]
    anomaly_data = data[labels == -1]
    plt.figure(figsize=(8, 6))
    plt.scatter(normal_data[:, 0], normal_data[:, 200], c='blue', label='Normal')
    plt.scatter(anomaly_data[:, 0], anomaly_data[:, 200], c='red', label='Anomaly')
    plt.legend()
    plt.title('Isolation Forest Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Removal of anomalous samples (isolation forests)
def IForest(data_D, ratio):
    # Construct mode
    model_IF = IsolationForest(n_estimators=100,
                               max_samples='auto',
                               contamination=float(ratio),
                               max_features=1.0)
    labels_IF = model_IF.fit_predict(data_D)
    # Visualize 3D data
    visualize_3D(data_D, labels_IF)
    data_IF = data_D[labels_IF == 1, :]
    return data_IF, labels_IF[labels_IF == 1]


# load & preprocessing
def data_get_1D(path, name, path_gt, name_gt, methods):
    preprocess_functions = {
        "SG": SG_smooth,
        "normalize": Normalization,
        "standardize": Standardization,
        "MSC": MSC,
        "SNV": SNV,
        "FD": FD
    }
    """
    Load and preprocess 1D spectral data for classification
    Args:
    path (str): Enter the path to the data file
    name (str): Enter the name of the data variable
    path_gt (str): The path to the truth data file
    name_gt (str): The name to the truth data file

    Returns:
    data_D (numpy.ndarray): Spectral data after preprocessing
    data_L (numpy.ndarray): Categorized Tags
    """
    # Load input and output (category labeled) images
    input_image = loadmat(path)[name]
    output_image = loadmat(path_gt)[name_gt]
    # Initialize lists to store data and labels
    data_list = []
    label_list = []
    # Iterate over the image
    for i in range(output_image.shape[0]):  # rows
        for j in range(output_image.shape[1]):  # columns
            if output_image[i][j] != 0:
                data_list.append(input_image[i][j])
                label_list.append(output_image[i][j])
    # Converting lists to numpy arrays
    # The original data_D is a three-dimensional array of shape (n_samples, n_bands, 1),
    # where n_samples denotes the number of samples and n_bands denotes the number of bands
    data_D = np.array([data_list])
    data_L = np.array(label_list)    # labels
    data_L = to_categorical(data_L)  # Converting labels to categorized form
    data_D = data_D.reshape([data_D.shape[1], data_D.shape[2]])  # Reshape spectral data (n_samples, n_bands)

    # preprocessing
    for method in methods:  # Use the corresponding preprocessing methods in the order in which they appear in the list
        if method in preprocess_functions:
            data_D = preprocess_functions[method](data_D)
        else:
            raise ValueError(f"Unsupported preprocessing method: {methods}")

    data_D = data_D.reshape([data_D.shape[0], data_D.shape[1], 1])  # Reshape data for CNN input (n_samples, n_bands, 1)

    return data_D, data_L
