# %%

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

tf.random.set_seed(1234)
np.random.seed(seed=1235)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# MNIST FASHION
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

X, y = ProcessMnistFashion.processMnistFashionData(train_images, train_labels)


# CALCULATE FP AND FN
# %%
def calc_fp(cm):
    if np.sum(cm[:,1])==0:
        fpr = 0
    else:
        fpr = (cm[0,1]/np.sum(cm[:,1]))
    return fpr

def calc_fn(cm):
    if np.sum(cm[:,0])==0:
        tnr = 0
    else:
        tnr = (cm[1,0]/np.sum(cm[:,0]))
    return tnr

# CLASSES FOR EXPERIMENTS
clsA = [2,2,0,5,6,3,5,8,7]
clsB = [4,6,3,7,4,7,9,2,9]

#  MODEL SIZES
modelSizes = [1,2,3,4,5]
# modelSizes = [1]

#  IMBALANCE RATIOS
irs = [0.05,0.15,0.3]

#  BATCH SIZE

btchSz = 32

# %% EXPERIMENT LOOP

inputDim = X[0].shape
outputDim = 2
fmResults = np.ndarray(shape=(0,2))
gmResults = np.ndarray(shape=(0,2))
baResults = np.ndarray(shape=(0,2))
fpResults = np.ndarray(shape=(0,2))
fnResults = np.ndarray(shape=(0,2))
conMat = []

for ir in irs:
    for trgClsIdx in range(len(clsA)):
        ssIdx = np.where(y==clsA[trgClsIdx])[0]
        tmpIdx = np.where(y==clsB[trgClsIdx])[0]
        tmpIdx = np.random.choice(tmpIdx, int(np.round(len(ssIdx)*ir)))
        ssIdx = np.append(ssIdx, tmpIdx)
        ssIdx = np.random.choice(ssIdx, len(ssIdx), replace=False)
        print(np.unique(y[ssIdx]))
        print(str(np.sum(y[ssIdx]==clsA[trgClsIdx]))+","+str(np.sum(y[ssIdx]==clsB[trgClsIdx])))
        X_exp = X[ssIdx]
        y_exp = y[ssIdx]
        y_exp[np.where(y_exp==clsA[trgClsIdx])[0]] = 0
        y_exp[np.where(y_exp==clsB[trgClsIdx])[0]] = 1
        #model depth loop
        for ms in modelSizes:
            #Stratified CV loop
            rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=36851234)
            tmpGm = np.array([])
            tmpBa = np.array([])
            tmpFm = np.array([])
            tmpFp = np.array([])
            tmpFn = np.array([])
            tmpCM = np.zeros(2*2).reshape(2,2)
            for train_index, test_index in rskf.split(X_exp, y_exp):
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
                #create model
                model = DeepModel.get_modelCnnSmall(inputDim, outputDim, hidden=10, depth=ms)
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=1e-5)
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_prc', verbose=0,patience=10,mode='max',restore_best_weights=True)
                model.fit(X_train2, y_trainEncoded, batch_size=btchSz, epochs=50, shuffle=True, validation_data=(X_val, y_valEncoded),verbose=0, callbacks=[early_stopping,reduce_lr])
                #predict
                y_prob = model.predict(X_test)
                y_pred = np.argmax(y_prob,axis=1)
                tmpFm = np.append(tmpFm, f1_score(y_test, y_pred, average='macro'))
                tmpGm = np.append(tmpGm, geometric_mean_score(y_test, y_pred))
                tmpBa = np.append(tmpBa, balanced_accuracy_score(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                tmpCM = tmpCM + cm
                tmpFp = np.append(tmpFp, calc_fp(cm))
                tmpFn = np.append(tmpFn, calc_fn(cm))
                model = None
                tf.keras.backend.clear_session()
            fmResults = np.concatenate((fmResults, np.array([np.mean(tmpFm), np.std(tmpFm)]).reshape(1,2)),axis=0)
            gmResults = np.concatenate((gmResults, np.array([np.mean(tmpGm), np.std(tmpGm)]).reshape(1,2)),axis=0)
            baResults = np.concatenate((baResults, np.array([np.mean(tmpBa), np.std(tmpBa)]).reshape(1,2)),axis=0)
            fpResults = np.concatenate((fpResults, np.array([np.mean(tmpFp), np.std(tmpFp)]).reshape(1,2)),axis=0)
            fnResults = np.concatenate((fnResults, np.array([np.mean(tmpFn), np.std(tmpFn)]).reshape(1,2)),axis=0)
            conMat.append(tmpCM)

np.savetxt("mnistFashionCnnFmeasure.csv", fmResults, delimiter=",")
np.savetxt("mnistFashionCnnGmean.csv", gmResults, delimiter=",")
np.savetxt("mnistFashionCnnBalAcc.csv", baResults, delimiter=",")
np.savetxt("mnistFashionCnnFalsePos.csv", fpResults, delimiter=",")
np.savetxt("mnistFashionCnnFalseNeg.csv", fnResults, delimiter=",")
np.savetxt("mnistFashionCnnConMat.csv", conMat, delimiter=",")


