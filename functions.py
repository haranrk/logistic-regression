import numpy as np
from matplotlib import pyplot as plt
import random as rd
import inspect
from functions import *
import idx2numpy as idx



#Weight Initialisation
def initialize_weights(number_of_features,val=10**-2):
	w=np.full(number_of_features,val)
	w=np.random.random(number_of_features)
	return w

def func(x,y,a=7,b=3):
	# return (y/a)**2+(x/b)**2-1
	return y-x+1
	# return y-x**2+4
	# return y-5*np.sin(x/10*np.pi)

#Creates the feature 
def create_features(x1,x2,level=0):
	if level==0:
		X = np.ones((len(x1),3))
		X[:,1]=x1
		X[:,2]=x2
	elif level==1:
		X = np.ones((len(x1),6))
		X[:,1]=x1
		X[:,2]=x2
		X[:,3]=x1**2
		X[:,4]=x2**2
		X[:,5]=x2*x1

	return X

def sigmoid(x):
	return 1/(1+np.exp(-x))

def batch_grad(X_training,y_training,alpha,grad_threshold):
	number_of_features=X_training.shape[1]
	w=initialize_weights(number_of_features,0)
	data_size=len(y_training)
	grad_w=np.full(number_of_features,11)
	L=np.array([1,2])
	count = 1
	print(X_training.shape)
	# while (abs(L[count]-L[count-1])>grad_threshold):
	while (L[count]>0.01):
		Y_training=sigmoid(np.dot(X_training,w.T))
		grad_w=np.dot(-1*2*(y_training-Y_training).T,X_training)/data_size
		w=w-alpha*grad_w
		count+=1
		Y_training=sigmoid(np.dot(X_training,w.T))
		val=sum(y_training*np.log(Y_training+0.0001)+(1-y_training)*np.log(1-Y_training+0.0001))/float(data_size)*-1
		L=np.append(L,val)
		# print(val,count)
		# if count%1000==0:
		# 	useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	# write_plot(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])	

def stochastic_grad(X_training,y_training,alpha,grad_threshold):
	number_of_features=X_training.shape[1]
	w=initialize_weights(number_of_features,0)
	data_size=len(y_training)
	grad_w=np.full(number_of_features,11)
	L=np.array([1,2])
	count = 1
	print(X_training.shape)
	# while (abs(L[count]-L[count-1])>grad_threshold):
	while (L[count]>0.01):
		Y_training=0
		for idx in range(0,data_size):
			Y_training=sigmoid(np.dot(X_training[idx,:],w.T))
			grad_w = np.dot(-1*(y_training[idx]-Y_training).T,X_training[idx,:])
			w=w-alpha*grad_w
		count+=1
		Y_training=sigmoid(np.dot(X_training,w.T))
		val=sum(y_training*np.log(Y_training+0.0001)+(1-y_training)*np.log(1-Y_training+0.0001))/float(data_size)*-1
		L=np.append(L,val)
		# print(val,count)
	# 	if count%1000==0:
	# 		useful_plots(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])
	# write_plot(L,count,X_training,y_training,w,inspect.getframeinfo(inspect.currentframe())[2])	
	return w

def useful_plots(L,count,X_training,y_training,w,algo_name,method=0):
	print("Iteration %d"%(count))
	print(L[count])
	plt.figure(1)
	plt.title(algo_name)
	plt.plot(range(0,len(L)),L)
	plt.xlabel("number of iterations")
	plt.ylabel("Error")
	if method==0:
		plt.figure(2)
		plt.clf()
		plot0(X_training[:,1],X_training[:,2],y_training)
		
		X=np.arange(-10,10,0.1)
		Y=np.arange(-10,10,0.1)
		X,Y=np.meshgrid(X,Y)
		X_1=create_features(X.reshape(-1),Y.reshape(-1))
		plt.contour(X,Y,np.dot(X_1,w.T).reshape(200,200),[0],colors="y")

		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.title(algo_name)
		plt.draw()
		plt.pause(0.0001)

	elif method==1:
		print("NA")

def write_plot(L,count,X_training,y_training,w,algo_name,method=0):
	print("Writing plots for %s" % (str(algo_name)))
	plt.figure(1)
	plt.clf()
	plt.title(algo_name)
	plt.plot(range(0,len(L)),L)
	plt.xlabel("number of iterations | Total iterations = %f"%(len(L)))
	plt.ylabel("Loss | Final loss %f"%(L[-2]))
	plt.savefig('graphs/%s_convergence.jpg'%(algo_name))
	if method==0:	
		plt.figure(2)
		plt.clf()
		plot0(X_training[:,1],X_training[:,2],y_training)
		X=np.arange(-10,10,0.1)
		Y=np.arange(-10,10,0.1)
		X,Y=np.meshgrid(X,Y)
		X_1=create_features(X.reshape(-1),Y.reshape(-1))
		plt.contour(X,Y,np.dot(X_1,w.T).reshape(200,200),[0],colors="y")
		plt.xlabel("x1")
		plt.ylabel("x2")
		plt.title(algo_name)
		plt.savefig('graphs/%s_hypothesis.jpg'%(algo_name))
	elif method==1:
		print("NA")

def sample_hypothesis(data_size):
	y = np.full(data_size,0.5)
	x1 = np.full(data_size,0.5)
	x2 = np.full(data_size,0.5)
	for i in range(0,data_size):
		x2[i] = float(rd.uniform(-10,10))
		x1[i] = float(rd.uniform(-10,10))
		if func(x1[i],x2[i])>0:
			y[i] = 1
		else:
			y[i] = 0
	create_features(x1,x2)
	return x1,x2,y



def import_mnist():
	test_imgs=idx.convert_from_file("t10k-images-idx3-ubyte")
	test_lbls=idx.convert_from_file("t10k-labels-idx1-ubyte")
	train_imgs=idx.convert_from_file("train-images-idx3-ubyte")  
	train_lbls=idx.convert_from_file("train-labels-idx1-ubyte")
	i=np.where((train_lbls==1)|(train_lbls==0))
	train_lbls=train_lbls[i]
	train_imgs=train_imgs[i]
	i=np.where((test_lbls==1)|(test_lbls==0))
	test_lbls=test_lbls[i]
	test_imgs=test_imgs[i]
	return train_imgs,train_lbls,test_imgs,test_lbls

def normalise_dataset(X):
	for idx in range(0,X.shape[1]):
		X[:,idx]=(X[:,idx]-np.mean(X[:,idx]))/np.std(X[:,idx])
		# X[:,idx]=X[:,idx]/np.std(X[:,idx])
	return X	

def plot0(x1,x2,y):
	X=np.arange(-10,10,0.1)
	Y=np.arange(-10,10,0.1)
	X,Y=np.meshgrid(X,X)
	plt.contour(X,Y,func(X,Y),[0])		

	for i in range(0,len(x1)):
		if y[i] == 1:
			plt.plot(x1[i],x2[i],'bo',markersize=1)
		else:
			plt.plot(x1[i],x2[i],'ro',markersize=1)
	plt.draw()
	plt.ylim([-10,10])
	plt.xlim([-10,10])

def calc_error(X_test,y_test,w):
	count=0
	for i,y in enumerate(y_test):
		if binary(sigmoid(np.dot(X_test[i,:],w)))==y:
			count+=1
	return count/len(y_test)

	return ((y_test-np.dot(X_test,w.T))**2).mean()

def binary(x):
	if x>0.5:
		return 1
	elif x<0.5:
		return 0

#Separate Dataset into training and test datasets
def separate_datasets(X,y,train_percent=0.8):
	train_indices = rd.sample(range(0,len(X)),int(train_percent*len(X)))
	test_indices = [i for i in range(0,len(X)) if i not in train_indices]
	x_training = np.ones((len(train_indices),X.shape[1]))
	x_test = np.ones((len(test_indices),X.shape[1]))
	y_test = np.ones(len(test_indices))
	y_training = np.ones(len(train_indices))
	for idx,i in enumerate(train_indices):
		x_training[idx,:]=X[i,:]
		y_training[idx]=y[i]

	for idx,i in enumerate(test_indices):
		x_test[idx,:]=X[i,:]
		y_test[idx]=y[i]

	return x_training,y_training,x_test,y_test