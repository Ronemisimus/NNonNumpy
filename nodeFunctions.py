import numpy as np

# these are all the functions to calculate forward and backward passes

# global documantaion:
# forward functions get a dictionary of the vectors they need to calculate their result
# this dictionary is children
# backwards functions get children too
# and they also get the gradients of their output and their output

# normal linear layer
def LinearLayerForward(children:dict):
    X:np.ndarray = children["X"].vec
    # add bias row to X
    X = np.append(X,np.ones(X.shape[:-1]+(1,)),axis=-1)
    w:np.ndarray = children["w"].vec
    # normal linear layer
    return X@w

# linear layer backpropogation
def LinearLayerBackward(children:dict,output:np.ndarray,output_grad):
    res = {}
    X:np.ndarray = children["X"].vec 
    w:np.ndarray = children["w"].vec # (batch,input,output)
    alpha:float = float(children["alpha"].vec)
    # add bias row to X
    X = np.append(X,np.ones(X.shape[:-1]+(1,)),axis=-1) # (batch,input,)
    
    # tansform it to a collumn of collumns (batch, input, 1)
    X = X[:,:,None]

    # output_grad[:,None,:] - turn it to collumn of rows (batch, 1, output)
    # X@output_grad - (batch, input, 1) @ (batch, 1, output) = (batch,input,output)
    # every collumn in X is doing np.outer with every row in output_grad
    # 2*w[None] - allows to add w to every w_grad in every sample for regularization
    res["w"] = X@(output_grad[:,None,:]) + 2*alpha*w[None]

    # w[None,:-1,:] - (1,input, output)
    # None - allows to multiply it with batched gradiant
    # :-1 - cuts out the bias gradiant
    # output_grad[:,:,None] - make it a collumn of collumns - (batch, output, 1)
    # (1,input, output) @ (batch, output, 1) = (batch, input, 1)
    # [:,:,0] - remove the last unneeded dimention: (batch, input, 1) -> (batch, input)
    res["X"] = (w[None,:-1,:]@output_grad[:,:,None])[:,:,0]

    # no gradient for regularization
    res["alpha"] = None

    return res

# normal relu
def ReLuForward(children:dict):
    X:np.ndarray = children["X"].vec
    X[X<0]=0
    return X

# relu back propogation
def ReLuBackward(children:dict,output:np.ndarray,output_grad):
    res = {}
    X:np.ndarray = children["X"].vec
    X[X<0]=0
    X[X>=0]=1
    if output_grad is not None:
        res["X"]=X*output_grad
    else:
        res["X"]=X
    return res

def softMaxForward(children:dict):
    X:np.ndarray = children["X"].vec
    
    # add stability
    m = X.max(axis=-1,keepdims=True)
    X = X-m
    
    X = np.exp(X)
    return X/np.sum(X,axis=-1,keepdims=True)

# softmax and CrossEntrophy united
def crossEntrophyLossForward(children:dict):
    X = children["X"].vec
    Y = children["Y"].vec
    # get softmax
    X = softMaxForward(children)
    
    # clip to low or to high values
    epsilon = 1e-12
    X = np.clip(X, epsilon, 1. - epsilon)
    
    # for every row in X (let's say k) and Y (let's say j) it does a K@J
    # then it does log and -
    # then an average across the batch
    return np.mean(-np.log(X[:,None,:]@Y[:,:,None]),axis=0)

def crossEntrophyLossBackward(children:dict,output:np.ndarray,output_grad=None):
    res = {}
    X:np.ndarray = children["X"].vec
    Y:np.ndarray = children["Y"].vec
    X = softMaxForward(children)
    # as calculated in the doc
    res["X"] = X - Y
    res["Y"] = None
    return res


def mseForward(children:dict):
    X = children["X"].vec
    Y = children["Y"].vec
    # simple suff
    return np.average(np.sum((X-Y)**2,axis=-1))

def mseBackward(children:dict,output:np.ndarray,output_grad=None):
    res = {}
    X:np.ndarray = children["X"].vec
    Y:np.ndarray = children["Y"].vec
    res["X"] = 2*(X-Y)
    res["Y"] = None
    return res


def sigmoidForward(children:dict):
    x = children["X"].vec
    return 1 / (1 + np.exp(-x))


def sigmoidBackward(children:dict,output:np.ndarray,output_grad=None):
    res = {}
    res["X"] = output*(1-output)*output_grad
    return res
