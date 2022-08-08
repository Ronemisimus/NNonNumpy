from math import sqrt
import numpy as np
from nodeFunctions import LinearLayerForward, LinearLayerBackward, sigmoidBackward, sigmoidForward, softMaxForward
from nodeFunctions import ReLuForward, ReLuBackward
from nodeFunctions import crossEntrophyLossForward, crossEntrophyLossBackward
from nodeFunctions import mseForward, mseBackward
import pickle

# a single node in the network tree
class VectorNode:
    def __init__(self,keep_grad,function_input,derivative_input):
        # result vector
        self.vec = None
        # wether to keep gradients for the vector
        self.keep_grad=keep_grad
        # the function to calculate vec from child nodes
        self.f_in = function_input
        # function to calculate derivative to child nodes
        self.derv_in = derivative_input
        # parent node
        self.parent = None
        # children with their role for the parent
        self.children = {}

        if self.keep_grad:
            self.grad = None

    def forward(self):
        self.vec = self.f_in(self.children)

    def backward(self,output_grad=None):
        if self.keep_grad:
            # save node gradient for weights
            self.grad = output_grad
        if self.derv_in is not None:
            # calculate derivatives
            grads:dict = self.derv_in(self.children,self.vec,output_grad)
            # pass each derivative to correct child recursivly
            for name, child in self.children.items():
                if grads[name] is not None:
                    child.backward(grads[name])

    def update(self, lr):
        # if it's a weight node
        if self.keep_grad:
            # get the size of the batch
            shape = np.array(self.grad.shape[:-2])
            # averege the batch
            for axis in range(len(shape)-1,-1,-1):
                self.grad = np.average(self.grad,axis=axis)
            # update the vector            
            self.vec = self.vec - lr*self.grad
            # zero the gradients
            self.grad = np.zeros_like(self.grad)
        # update child nodes recursivly
        for name, child in self.children.items():
            child.update(lr)
        
    # recursivly save best weights
    def pocket(self):
        if self.keep_grad:
            self.best = self.vec.copy()
        for name, child in self.children.items():
            child.pocket()
    
    # recursivly reset best weights
    def best_weights(self):
        if self.keep_grad:
            self.vec= self.best
        for name, child in self.children.items():
            child.best_weights()

    # get the network weights
    def parameters(self):
        res = []
        for child in self.children.values():
            if child.keep_grad:
                res.append(child.vec)
            else:
                res.extend(child.parameters())
        return res

    # set the network weights from the list
    def set_parameters(self,params, idx):
        for child in self.children.values():
            if child.keep_grad:
                child.vec = params[idx]
                idx = idx + 1
            else:
                idx = child.set_parameters(params, idx)
        return idx

    # save the network parameters to file
    def save_state(self,name:str):
        params = self.parameters()
        with open(name,"wb") as f:
            pickle.dump(params,f)
    
    # load the network parameters to file
    def load_state(self,name):
        with open(name,"rb") as f:
            params = pickle.load(f)
            self.set_parameters(params,0)

# a node that only contains a vector - usually weights or inputs
class EdgeNode(VectorNode):
    def __init__(self,keep_grad,vec):
        self.vec = vec
        self.keep_grad=keep_grad
        self.f_in = None
        self.derv_in = None
        self.parent = None
        self.children = {}

# a linear node
class LinearLayerNode(VectorNode):
    def __init__(self,input_dim,output_dim,alpha:float,child_input:VectorNode=None):
        # should know how to act as a linear transform
        super().__init__(False,LinearLayerForward,LinearLayerBackward)
        # set the children
        if child_input is not None:
            # input of the layer
            self.children["X"] = child_input
            child_input.parent = self
        # weights are set to have a small normal
        w = np.random.normal(loc=0,scale=2/sqrt((input_dim+1)*output_dim),size=(input_dim+1,output_dim))
        # and added as child of the node
        self.children["w"] = EdgeNode(True,w)
        # alpha is also a child
        self.children["alpha"] = EdgeNode(False,np.array([alpha]))

    def __call__(self, X:np.ndarray=None):
        if X is not None:
            # connect the input of the layer
            self.children["X"] = EdgeNode(False,X)
        # run the layer
        self.forward()

# relu layer
class ReLu(VectorNode):
    def __init__(self,childNode:VectorNode):
        super().__init__(False, ReLuForward, ReLuBackward)
        # must have a child X
        self.children["X"] = childNode
        childNode.parent = self

    def __call__(self):
        self.forward()

# softmax cross entrophy layer
class CrossEntrophyLoss(VectorNode):
    def __init__(self, childX:VectorNode):
        super().__init__(False, crossEntrophyLossForward, crossEntrophyLossBackward)
        self.children["X"] = childX
        childX.parent = self

    def __call__(self, Y:np.ndarray):
        # Y should be a one hot vector for this calculation
        Y = Y.astype(np.int32)
        Y = np.eye(self.children["X"].vec.shape[-1])[Y]
        # add Y as child
        self.children["Y"] = EdgeNode(False,Y)
        self.forward()

# mse layer
class MSE(VectorNode):
    def __init__(self, childX:VectorNode):
        super().__init__(False, mseForward, mseBackward)
        # child is a vector of predictions
        self.children["X"] = childX
        childX.parent = self

    def __call__(self, Y:np.ndarray):
        # Y must be a one hot vector
        Y = Y.astype(np.int32)
        Y = np.eye(self.children["X"].vec.shape[-1])[Y]
        # add Y as child
        self.children["Y"] = EdgeNode(False,Y)
        self.forward()
        
# sigmoid layer
class Sigmoid(VectorNode):
    def __init__(self, childX:VectorNode):
        super().__init__(False, sigmoidForward, sigmoidBackward)
        self.children["X"] = childX
        childX.parent = self
    
    def __call__(self):
        self.forward()
