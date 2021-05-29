
from ProcessMnistFashion import processMnistFashionData
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import geometric_mean_score
import DeepModel
import ProcessMnistFashion
from sklearn.utils.class_weight import compute_class_weight
import datetime
from copy import copy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# MNIST FASHION
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

X, y = ProcessMnistFashion.processMnistFashionData(train_images, train_labels)

X_indTest, y_indTest = ProcessMnistFashion.processMnistFashionData(train_images, train_labels)


plt.rcParams.update({'font.size': 16})
names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
clsA = [2,2,0,5,6,3,5,8,7]
clsB = [4,6,3,7,4,7,9,2,9]
modelSizes = [1,2,3,4,5]
#  IMBALANCE RATIOS
irs = [0.05,0.15,0.3]

#  BATCH SIZE
btchSz = 32

cnnOut = [4,11,19]
fcOut = [6,13,21]

for i in range(len(clsA)):
    for j in range(len(modelSizes)):
        ms = modelSizes[j]
        cnnOutLayer = cnnOut[j]
        fcOutLayer = fcOut[j]
        np.random.seed(seed=486)
        ssIdx = np.where(y==c1)[0]
        tmpIdx = np.where(y==c2)[0]
        tmpIdx = np.random.choice(tmpIdx, int(np.round(len(ssIdx)*irs[0])))
        ssIdx = np.append(ssIdx, tmpIdx)
        ssIdx = np.random.choice(ssIdx, len(ssIdx), replace=False)
        X_exp = X[ssIdx]
        y_exp = y[ssIdx]
        y_exp[np.where(y_exp==c1)[0]] = 0
        y_exp[np.where(y_exp==c2)[0]] = 1

        #Stratified CV loop
        rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=36851234)
        for train_index, test_index in rskf.split(X_exp, y_exp):
            print("...")

        inputDim = X_exp[0].shape
        outputDim = 2
        X_train, X_test = X_exp[train_index, :], X_exp[test_index, :]
        y_train, y_test = y_exp[train_index], y_exp[test_index]
        ss = StratifiedShuffleSplit(n_splits=2, test_size=0.3, random_state=568453)
        val_index, _ = ss.split(X_train, y_train)
        X_train2, X_val = X_train[val_index[0]], X_train[val_index[1]]
        y_train2, y_val = y_train[val_index[0]], y_train[val_index[1]]
        y_trainEncoded = tf.keras.utils.to_categorical(y_train2)
        y_valEncoded = tf.keras.utils.to_categorical(y_val)
        y_testEncoded = tf.keras.utils.to_categorical(y_test)
        y_trainEncoded = tf.keras.utils.to_categorical(y_train2)

        model = DeepModel.get_modelCnnSmall(inputDim, outputDim, hidden=10, dropout=[3,4,5,6,7], depth=modelSizes[0])
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=0,patience=10,mode='max',restore_best_weights=True)
        model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=100, shuffle=True, validation_data=(X_val, y_valEncoded),verbose=0, callbacks=[early_stopping,reduce_lr])

        #plot after cnn blocks
        inter_output_model = tf.keras.Model(model.input, outputs=model.get_layer(index = cnnOutLayer).output)
        inter_output = inter_output_model.predict(X_indTest)

        modelTsne = TSNE(n_components=2, random_state=0)
        tsne = modelTsne.fit_transform(inter_output)

        idxA = np.where(y_indTest==clsA[0])[0]
        plt.scatter(x=tsne[idxA,0],y=tsne[idxA,1], s=35, alpha=0.5, label=names[clsA[0]]+" ("+str(clsA[0])+")")
        idxB = np.where(y_indTest==clsB[0])[0]
        plt.scatter(x=tsne[idxB,0],y=tsne[idxB,1], s=35, alpha=0.5, label=names[clsB[0]]+" ("+str(clsB[0])+")")
        plt.legend()
        plt.savefig("mfEmbeddedAfterCnnModelSize"+str(ms)+"Class"+str(clsA[i])+'versus'+str(clsB[i])+".pdf")
        plt.close()

        #plot after after dense
        inter_output_modelFc = tf.keras.Model(model.input, outputs=model.get_layer(index = 25).output)
        inter_outputFc = inter_output_modelFc.predict(X_indTest)

        modelTsne = TSNE(n_components=2, random_state=0)
        tsne = modelTsne.fit_transform(inter_outputFc)

        idxA = np.where(y_indTest==clsA[0])[0]
        plt.scatter(x=tsne[idxA,0],y=tsne[idxA,1], s=35, alpha=0.5, label=names[clsA[0]]+" ("+str(clsA[0])+")")
        idxB = np.where(y_indTest==clsB[0])[0]
        plt.scatter(x=tsne[idxB,0],y=tsne[idxB,1], s=35, alpha=0.5, label=names[clsB[0]]+" ("+str(clsB[0])+")")
        plt.legend()
        plt.savefig("mfEmbeddedAfterDenseModelSize"+str(ms)+"Class"+str(clsA[i])+'versus'+str(clsB[i])+".pdf")
        plt.close()