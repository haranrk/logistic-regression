from functions import *
import numpy as np

train_imgs,y_training,test_imgs,y_test=import_mnist()
#Without Normalising
X_training=train_imgs.reshape(-1,784)
X_test=test_imgs.reshape(-1,784)
w=stochastic_grad(X_training,y_training,10**-2,0.01)
print(calc_error(X_test,y_test,w))

#With Normalising
X_training=train_imgs.reshape(-1,784)/255
X_test=test_imgs.reshape(-1,784)/255
w=stochastic_grad(X_training,y_training,10**-2,0.01)
print(calc_error(X_test,y_test,w))
plt.pause(4)