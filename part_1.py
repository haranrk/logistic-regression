from functions import *

plt.ion()
data_size=1000
alpha=10**-2

x1,x2,y=sample_hypothesis(data_size)
X=create_features(x1,x2)
X_training,y_training,X_test,y_test=separate_datasets(X,y)
w = stochastic_grad(X_training,y_training,10**-2,0.0001)
# w = batch_grad(X,y,10**-1,0.0001)
print(calc_error(X_test,y_test,w))
plt.pause(4)
plt.close()