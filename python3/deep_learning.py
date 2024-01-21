import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
import scipy.io
from keras.utils import to_categorical
import math
from keras import layers
from keras.layers import Conv1D, Dense, Dropout, BatchNormalization, MaxPooling1D, Activation, Flatten, Input
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.metrics import accuracy_score, confusion_matrix
import func_pre
import seaborn as sns


def pca_dr(src, lbl):
    # Perform PCA analysis
    pca = PCA(n_components=0.98)
    # Cumulative Variance Visualization
    pca.fit(src)
    component = pca.components_  # Getting Principal Components
    var = pca.explained_variance_ratio_  # Obtain the proportion of variance explained corresponding to each principal component
    cumulative_var = [sum(var[:i + 1]) for i in range(
        len(var))]  # Iteratively calculate the sum of the elements from the previous step to obtain the proportion of cumulative variance explained
    plt.figure()
    plt.plot(np.arange(len(var)), cumulative_var)

    # Transform data into principal component space (visualization)
    img_pc = pca.fit_transform(src)
    # Get the first three principal components
    pc1 = img_pc[:, 0]
    pc2 = img_pc[:, 1]
    pc3 = img_pc[:, 2]
    # Get the colors of different categories
    unique_labels = np.unique(lbl)
    colors = ['g', 'y', 'b', 'm', 'r']
    color_map = dict(zip(unique_labels, colors))
    # Plot 3D Scatter Plots
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for label in unique_labels:
        indices = np.where(lbl == label)[0] - 1
        ax.scatter(pc1[indices], pc2[indices], pc3[indices], c=color_map[label], label=str(label))
    # Setting Axis Labels
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend()
    plt.show()
    return img_pc, component.T


'''=====================attention mechanism====================='''


def CAM(inputs, b, gamma):
    # Number of channels of input feature map
    in_channel = inputs.shape[-1]
    # Calculate the adaptive convolution kernel size
    kernel_size = int(abs((math.log(in_channel, 2) + b) / gamma))
    # Convolutional kernel resizing
    if kernel_size % 2:
        kernel_size = kernel_size
    else:
        kernel_size = kernel_size + 1
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(inputs)
    x = layers.Reshape(target_shape=(in_channel, 1))(x)
    x = layers.Conv1D(filters=1, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = tf.nn.sigmoid(x)
    x = layers.Reshape((1, in_channel))(x)
    outputs = layers.multiply([inputs, x])
    return outputs


'''The following data in .mat format are the spectral features (or feature band indexes) 
extracted by MATLAB from the ROIs of train sets'''

# ===full-band===
data_D = scipy.io.loadmat(".../CNNinput5.mat")
data_D = data_D['data_stand']

labels1 = 0 * np.ones(38821)
labels2 = 1 * np.ones(44499)
labels3 = 2 * np.ones(45022)
labels4 = 3 * np.ones(56025)
labels5 = 4 * np.ones(43041)
labels = np.concatenate((labels1, labels2, labels3, labels4, labels5))
data_LB = to_categorical(labels)

'''Removal of anomalous samples (isolation forests)'''
data_D = np.reshape(data_D, (data_D.shape[0], data_D.shape[1]))
data_D, ttt = pca_dr(data_D, labels)
data1_D = data_D[:38821, :]
data2_D = data_D[38821:83320, :]
data3_D = data_D[83320:128342, :]
data4_D = data_D[128342:184367, :]
data5_D = data_D[184367:, :]
data1_D, data1_L = func_pre.IForest(data1_D, 0.01)
data2_D, data2_L = func_pre.IForest(data2_D, 0.01)
data3_D, data3_L = func_pre.IForest(data3_D, 0.01)
data4_D, data4_L = func_pre.IForest(data4_D, 0.01)
data5_D, data5_L = func_pre.IForest(data5_D, 0.01)
data_D = np.concatenate((data1_D, data2_D, data3_D, data4_D, data5_D))
data_D = np.reshape(data_D, (data_D.shape[0], data_D.shape[1], 1))
data_L_IF = np.concatenate((0 * data1_L, 1 * data2_L, 2 * data3_L, 3 * data4_L, 4 * data5_L))
data_LB = to_categorical(data_L_IF)  # Convert tags into categorized form

# ===PCA===
proj_pca = scipy.io.loadmat(".../proj_pca.mat")
proj_pca = proj_pca['proj_pca']
data_D = np.reshape(data_D, (data_D.shape[0], data_D.shape[1]))
data_PCA = np.dot(data_D, proj_pca)
data_D = np.reshape(data_D, (data_D.shape[0], data_D.shape[1], 1))
data_PCA = np.reshape(data_PCA, (data_PCA.shape[0], data_PCA.shape[1], 1))
# ===iRF===
bands_irf = scipy.io.loadmat(".../bands_irf.mat")
bands_irf = bands_irf['bands_iRF'] - 1
bands_irf = bands_irf.reshape(-1)
data_iRF = data_D[:, bands_irf, :]
# ===SPA===
bands_spa = scipy.io.loadmat(".../bands_spa.mat")
bands_spa = bands_spa['bands_spa'] - 1
bands_spa = bands_spa.reshape(-1)
data_SPA = data_D[:, bands_spa, :]

train_D, test_D, train_L, test_L = train_test_split(data_SPA, data_LB, test_size=0.2, random_state=42)

save_path = f"1DCNN.h5"
classes = 5

batchsize = 80
epochs = 60
num_classes = 5

input_shape = train_D.shape[1:]
# Define input layers and determine input dimensions
input = Input(shape=input_shape)
'''==========================CNN architecture=========================='''
# CNN parameters are determined by trial and error method
# C1
x = Conv1D(filters=6, kernel_size=4, padding='valid', input_shape=input_shape, kernel_regularizer=l2(0.001))(input)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2)(x)
# C2
x = Conv1D(filters=12, kernel_size=4, padding='valid')(x)
x = Dropout(0.1)(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=2)(x)
# C3
x = Conv1D(filters=24, kernel_size=3, padding='valid')(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=1)(x)
# C4
x = Conv1D(filters=48, kernel_size=4, padding='valid')(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=1)(x)
# channel attention mechanism
eca = CAM(x, 1, 2)
x = layers.add([x, eca])
# C5
x = Conv1D(filters=96, kernel_size=3, padding='valid')(x)
x = Activation('relu')(x)
x = MaxPooling1D(pool_size=1)(x)

x = Flatten()(x)

x = Dense(units=256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dense(units=128, activation='relu')(x)

output = Dense(units=num_classes, activation='softmax')(x)
model = Model(inputs=input, outputs=output)
model.compile(optimizer=Adam(lr=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
model.summary()
initial_weights = model.get_weights()

'''---------------------------------test---------------------------------'''
# accuracy
# ===full-band===
data_D_test = scipy.io.loadmat(".../Model_test.mat")
data_D_test = data_D_test['data_stand']
# data_D_test = np.reshape(data_D_test, (data_D_test.shape[0], data_D_test.shape[1], 1))
# ===PCA===
data_PCA_test = np.dot(data_D_test, proj_pca)
# ===iRF===
data_iRF_test = data_D_test[:, bands_irf]
# ===SPA===
data_SPA_test = data_D_test[:, bands_spa]

label1 = 0 * np.ones(5019)
label2 = 1 * np.ones(6304)
label3 = 2 * np.ones(6631)
label4 = 3 * np.ones(6251)
label5 = 4 * np.ones(5670)
data_L_test = np.concatenate((label1, label2, label3, label4, label5))
data_L_test = to_categorical(data_L_test)

'''---------------------------------Mask Visualization---------------------------------'''
mask_test = scipy.io.loadmat(".../mask_test_new.mat")
mask_test = mask_test['data_stand']
mask_PCA_test = np.dot(mask_test, proj_pca)
mask_iRF_test = mask_test[:, bands_irf]
mask_SPA_test = mask_test[:, bands_spa]

test_accuracies = []
for i in range(10):
    model.set_weights(initial_weights)
    print(f'=============={i}==============')
    history = model.fit(train_D, train_L, epochs=epochs, batch_size=batchsize, validation_data=(test_D, test_L))
    # Predictive labeling
    DATA = data_SPA_test
    DATA = np.reshape(DATA, (DATA.shape[0], DATA.shape[1], 1))
    pre_L = model.predict(DATA)
    predict_labels = np.argmax(pre_L, axis=1)

    MASK = mask_SPA_test
    MASK = np.reshape(MASK, (MASK.shape[0], MASK.shape[1], 1))
    mask_pre = model.predict(MASK)
    mask_pre = np.argmax(mask_pre, axis=1)
    mask_pre = mask_pre + 1

    OA_test = accuracy_score(np.argmax(data_L_test, axis=1), predict_labels)
    test_accuracies.append(OA_test)

print('**********Test Accuracy**********')
print(test_accuracies)

DATA = data_D_test
DATA = np.reshape(DATA, (DATA.shape[0], DATA.shape[1], 1))
# Predictive labeling
pre_L = model.predict(DATA)
predict_labels = np.argmax(pre_L, axis=1)
OA_test = accuracy_score(np.argmax(data_L_test, axis=1), predict_labels)
con_test = confusion_matrix(np.argmax(data_L_test, axis=1), predict_labels)
scipy.io.savemat(f'predict_labels.mat', mdict={'predict_labels': predict_labels})  # 保存
print('**********Test Accuracy**********')
print(f"{OA_test * 100}%")
print('**********Confusion Matrix**********')
print(con_test)

# === Visualization of Confusion Matrix  ===
con_mat_norm = con_test.astype('float') / con_test.sum(axis=1)[:, np.newaxis]  # normalize
con_mat_norm = np.around(con_mat_norm, decimals=4)
# === plot ===
plt.figure()
sns.heatmap(con_mat_norm, annot=True, fmt='.1%', cmap='GnBu')
plt.xlabel('Predicted labels', fontsize=12)
plt.ylabel('True labels', fontsize=12)
plt.show()

MASK = mask_test
MASK = np.reshape(MASK, (MASK.shape[0], MASK.shape[1], 1))
mask_pre = model.predict(MASK)
mask_pre = np.argmax(mask_pre, axis=1)
mask_pre = mask_pre + 1
scipy.io.savemat(f'predict_labels2.mat', mdict={'predict_labels2': mask_pre})  # 保存

# *'predict_labels' is used for evaluation of the model;
# *'predict_labels2' is used for visualization of the results in MATLAB.
