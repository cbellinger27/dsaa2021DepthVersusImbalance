# %%
import tensorflow as tf
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#plot TSNE of CIFAR 10, binary, multi-class, balanced, imbalanced

# %%

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
y = train_labels[..., np.newaxis].copy()
X = train_images.reshape((train_images.shape[0], 28, 28, 1))


# %%
plt.rcParams.update({'font.size': 16})
names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
clsA = [2,2,0,5,6,3,5,8,7]
clsB = [4,6,3,7,4,7,9,2,9]
for i in range(len(clsA)):
    c1 = clsA[i]
    c2 = clsB[i]
    ssIdx = np.where(y==c1)[0]
    ssIdx = np.append(ssIdx, np.where(y==c2)[0])
    print(np.unique(y[ssIdx]))

    y_ssTrain = y[ssIdx]
    X_ssTrain = X[ssIdx]

    #loop over each class label and sample 400 random images over each label and save the idx to subset
    np.random.seed(seed=486)
    ssIdx2=np.empty(0,dtype="int8")
    for c in np.unique(y_ssTrain):
        ssIdx2=np.append(ssIdx2,np.random.choice(np.where(y_ssTrain==c)[0],400,replace=False))

    X_ssTrain= X_ssTrain[ssIdx2]
    y_ssTrain= y_ssTrain[ssIdx2]

    #tsne embedding
    model = TSNE(n_components=2, random_state=0)
    tsne = model.fit_transform(X_ssTrain.reshape((len(X_ssTrain),28*28)))

    #plot tsne embedding
    idxA = np.where(y_ssTrain[:,0]==c1)[0]
    plt.scatter(x=tsne[idxA,0],y=tsne[idxA,1], s=35, alpha=0.5, label=names[clsA[i]]+" ("+str(clsA[i])+")")
    idxB = np.where(y_ssTrain[:,0]==c2)[0]
    plt.scatter(x=tsne[idxB,0],y=tsne[idxB,1], s=35, alpha=0.5, label=names[clsB[i]]+" ("+str(clsB[i])+")")
    plt.legend()
    plt.savefig("mfBalanced"+str(clsA[i])+'versus'+str(clsB[i])+".pdf")
    plt.close()

