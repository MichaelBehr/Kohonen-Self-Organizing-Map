# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 15:23:18 2020

@author: Michael
"""

import numpy as np
import matplotlib.pyplot as plt

########################## CLASS FOR KSON MODEL ##############################

# we are going to create a class to create our KOM model

class KSON:
    def __init__(self, input_data, sigma0,initial_weights):
        
        # store the input data, and output_labels
        self.input_data = input_data
        self.W = initial_weights
        self.sigma0 = sigma0
        self.current_epoch = 0
        
        # we use this for vectorizing our computations later
        self.Idx = np.zeros([100,100,2])
        for i in range(100):
            for j in range(100):
                self.Idx[i,j,:] = [i,j]


    # different neighborhood functions depending on which computation method
    # loop method
    def Neighborhood(self,x,g):
        return(np.exp(-self.euclidean_distance2(x,g)**2/(2*self.sigma()**2)))
        
    # vectorized
    def Neighborhood2d(self,x):
        return(np.exp(-self.euclidean_distance2d(x)**2/(2*self.sigma()**2)))
    
    # vectorized
    def euclidean_distance3d(self,x_input):
        d = (self.W - x_input)**2
        d = np.sqrt(d[:,:,0] + d[:,:,1] + d[:,:,2])
        return(d)
     
    # vectorized
    def euclidean_distance2d(self,x_input):
        d = (self.Idx - x_input)**2
        d = np.sqrt(d[:,:,0] + d[:,:,1])
        return(d)
    
    # loop
    def euclidean_distance2(self,p,q):
        return(np.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2))
        
# =============================================================================
# Original loop form of find_closest function
#   def find_closest(self,x_input):
#         x = 0
#         y = 0
#         min_dist = self.euclidean_distance(x_input,self.W[0,0,:])
#         d = np.zeros([100,100])
#         for i in range(len(self.W[:,0,0])):
#             for j in range(len(self.W[0,:,0])):
#                 d[i,j] = self.euclidean_distance(x_input,self.W[i,j,:])
#                 if(self.euclidean_distance(x_input,self.W[i,j,:]) < min_dist):
#                     x = i
#                     y = j
#                     min_dist = self.euclidean_distance(x_input,self.W[i,j,:])
#         print("closest match found: (" + str(x) + "," + str(y) + ")")
#         return(x,y)
# =============================================================================
        
    def find_closest(self,x_input):
        d = self.euclidean_distance3d(x_input)
        xy = np.unravel_index(np.argmin(d, axis=None), d.shape)
        #print("closest match found: (" + str(xy[0]) + "," + str(xy[1]) + ")")
        return(xy[0],xy[1])
        
    
# =============================================================================
# Original loop form for weight update function
#    def update_weights(self):
#         for i in range(len(self.input_data)):
#             # calculate winning neuron coordinates
#             x,y = self.find_closest(self.input_data[i])
#             for m in range(len(self.W[:,0,0])):
#                 for n in range(len(self.W[0,:,0])):
#                     self.W[m,n,:] = self.W[m,n,:] + self.alpha()*self.Neighborhood([x,y],[m,n])*(self.input_data[i]-self.W[m,n,:])
# =============================================================================
     
    def update_weights(self):
            for i in range(len(self.input_data)):
                # calculate winning neuron coordinates
                x,y = self.find_closest(self.input_data[i])
                xy = [x,y]
                W1 = np.multiply(self.Neighborhood2d(xy),(self.input_data[i]-self.W)[:,:,0])
                W2 = np.multiply(self.Neighborhood2d(xy),(self.input_data[i]-self.W)[:,:,1])
                W3 = np.multiply(self.Neighborhood2d(xy),(self.input_data[i]-self.W)[:,:,2])
                delta_W = np.dstack([W1,W2])
                delta_W = np.dstack([delta_W,W3])
                self.W = self.W + self.alpha()*delta_W
                            
                          
    def sigma(self):
        return(self.sigma0 * np.exp(-self.current_epoch/1000))
        
    def alpha(self):
        return(0.8*np.exp(-self.current_epoch/1000))
    
    def train(self,epoch):
        print("#### Training model with Sigma0 = ", self.sigma0)
        for epoch in range(epoch):
            self.update_weights()
            if ((epoch+1) == 20 or (epoch+1) == 40 or (epoch+1) == 100 or (epoch+1) == 1000):
                fig,ax = plt.subplots()
                ax.set(title='Sigma ='+str(self.sigma0)+' at Epoch = ' + str(epoch))
                plt.imshow(self.W)
                fig.savefig("Q4_"+str(self.sigma0)+"_epoch"+str(epoch+1)+".png")
            self.current_epoch = epoch

################################## SCRIPTS ####################################
# Red, Lime, Blue, Cyan, Magenta, Green + shade variants
Input_colours = [[255,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,255],[0,128,0], \
                 [128,0,0],[178,34,34],[165,42,42],[255,255,0],[154,205,50],[173,255,47], \
                 [0,206,209],[64,224,208],[127,255,212],[138,43,226],[139,0,139],[153,50,204], \
                 [219,112,147],[255,105,180],[255,192,203],[255,250,205],[255,255,224],[65,105,225]]

Input_colours = np.array(Input_colours)/255

# Create initial weight colour map and plot
initial_weights = np.random.rand(100,100,3)
fig = plt.figure()
plt.imshow(initial_weights)

##### Now we train multiple models (different sigma0)

########## Sigma0 = 1
sigma0 = 1
K_0 = KSON(Input_colours,sigma0,initial_weights)

# Train the model on the input data over 1000 epochs
K_0.train(1000)

########## Sigma0 = 10
sigma0 = 10
K_10 = KSON(Input_colours,sigma0,initial_weights)

# Train the model on the input data over 1000 epochs
K_10.train(1000)

########## Sigma0 = 30
sigma0 = 30
K_30 = KSON(Input_colours,sigma0,initial_weights)

# Train the model on the input data over 1000 epochs
K_30.train(1000)

########## Sigma0 = 50
sigma0 = 50
K_50 = KSON(Input_colours,sigma0,initial_weights)

# Train the model on the input data over 1000 epochs
K_50.train(1000)

########## Sigma0 = 70
sigma0 = 70
K_70 = KSON(Input_colours,sigma0,initial_weights)

# Train the model on the input data over 1000 epochs
K_70.train(1000)
