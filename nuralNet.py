from VectorNode import MSE, LinearLayerNode, ReLu, Sigmoid, CrossEntrophyLoss
from nodeFunctions import softMaxForward
import numpy as np
from tqdm import tqdm
from VectorNode import EdgeNode
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# general nuralnet
class nuralNet():
    def __init__(self,lr=1e-3,checkpoint_name="checkpoint"):
        # a list of objects to activate in series
        self.steps = []
        # the learning rate
        self.lr=lr
        # where to save the best progress
        self.checkpoint_name = checkpoint_name

    # predict function
    def __call__(self,X):
        prediction = None
        for i,step in enumerate(self.steps):
            if i==0:
                step(X)
            # not do the loss step
            elif i + 1 < len(self.steps):
                step()
        # get the last step runned result
        prediction = self.steps[-2].vec
        return prediction
    
    def predict(self,X):
        if len(X.shape)==1:
            # single example
            return self(X[None])
        else:
            # batch
            return self(X)
    
    def fit(self,X,Y,epochs=100,batch_size=32):
        # split to train and validation
        X, Xv, Y, Yv = train_test_split(X,Y,test_size=0.1,random_state=0)
        # split must be the same so we don't train on previous validation
        best_val_loss = 100
        # start training
        for epoch in range(epochs):
            print("epoch",epoch+1,"/",epochs)
            train_avg_loss, trin_losses = self.predictionAndLoss(X,Y,batch_size,train=True,shuffle=True)# train epoch
            val_avg_loss, val_losses = self.predictionAndLoss(Xv,Yv,batch_size,train=False,shuffle=False)# validation epoch
            # save weights on improvement
            if val_avg_loss < best_val_loss:
                self.steps[-1].pocket()
                self.save(self.checkpoint_name)
                best_val_loss = val_avg_loss
                print("saved state at epoch",epoch+1)
            print("train_loss:", round(train_avg_loss,5), "validation loss", round(val_avg_loss,5))
        self.steps[-1].best_weights()

    # single epoch
    def predictionAndLoss(self,X,Y,batch_size, train:bool,shuffle:bool=True):
        loss_list = []
        if shuffle:
            # create a random permutation
            perm = np.random.permutation(Y.shape[0])
        else:
            # create a series for non-shuffled run
            perm = range(Y.shape[0])
        # run progress bar
        for sample in (pbar:= tqdm(range(0,Y.shape[0],batch_size))):
            # get the batch size
            curr_size = min(Y.shape[0]-sample,batch_size)
            # get a list of batch indexes from permutation
            batch_idx = perm[sample:sample+curr_size]
            # run the model
            self(X[batch_idx,:])
            # calculate loss
            self.steps[-1](Y[batch_idx])
            # add loss to list
            loss_list.append(float(self.steps[-1].vec))
            
            # if training epoch we backpropogate and update weights
            if train:
                self.steps[-1].backward()
                self.steps[-1].update(self.lr)
            # show batch loss
            pbar.set_description("loss: "+str(round(loss_list[-1],5)))
        # return averege epoch loss
        return np.average(loss_list), loss_list
    
    def save(self,name):
        self.steps[-1].save_state(name)
    
    def load(self,name):
        self.steps[-1].load_state(name)

    def score(self,Xt,Yt):
        # get prediction
        pred = np.argmax(self(Xt),axis=-1)
        print(classification_report(Yt,pred))
        # return the error rate
        return np.sum(pred!=Yt)/Yt.shape[0]

class SoftMaxNet(nuralNet):
    # layers - list of numbers, each number represents a hudden layer, first and last represent input and output layer
    def __init__(self, layers: list, lr=0.001, alpha=0.001):
        super().__init__(lr)
        for layer_idx in range(len(layers)-1):
            # each layers input is the previous layer
            child_input = None if len(self.steps)==0 else self.steps[-1]
            # add a linear transition from current hidden layer to next layer
            self.steps.append(LinearLayerNode(layers[layer_idx],layers[layer_idx+1],alpha=alpha,child_input=child_input))
            # the last layer might not need an activation - in this case the activation is 
            # softmax and it doesn't have it's own layer
            if layer_idx < len(layers)-2:
                # for every transition add a relu activation
                self.steps.append(ReLu(self.steps[-1]))
        # add a loss function
        self.steps.append(CrossEntrophyLoss(self.steps[-1]))
    
    # predictions also applies softmax
    def __call__(self, X):
        pred = super().__call__(X)
        children = {"X":EdgeNode(False,pred)}
        return softMaxForward(children)

# same net with sigmoid activations - not better
class SoftMaxNet2(nuralNet):
    def __init__(self, layers: list, lr=0.001, alpha=0.001):
        super().__init__(lr)
        for layer_idx in range(len(layers)-1):
            child_input = None if len(self.steps)==0 else self.steps[-1]
            self.steps.append(LinearLayerNode(layers[layer_idx],layers[layer_idx+1],alpha=alpha,child_input=child_input))
            if layer_idx < len(layers)-2:
                self.steps.append(Sigmoid(self.steps[-1]))
        #self.steps.append(Sigmoid(self.steps[-1]))
        self.steps.append(CrossEntrophyLoss(self.steps[-1]))

    # predictions also applies softmax
    def __call__(self, X):
        pred = super().__call__(X)
        children = {"X":EdgeNode(False,pred)}
        return softMaxForward(children)

# an mse based net
class MSENet(nuralNet):
    def __init__(self,layers:list, lr=0.001, alpha=0.001):
        super().__init__(lr)
        for layer_idx in range(len(layers)-1):
            child_input = None if len(self.steps)==0 else self.steps[-1]
            # add linear transitions
            self.steps.append(LinearLayerNode(layers[layer_idx],layers[layer_idx+1],alpha=alpha,child_input=child_input))
            if layer_idx < len(layers)-2:
                # add sigmoid activations
                self.steps.append(Sigmoid(self.steps[-1]))
        # add sigmoid after last layer
        self.steps.append(Sigmoid(self.steps[-1]))
        # add loss function
        self.steps.append(MSE(self.steps[-1]))