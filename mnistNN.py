from os.path import isfile
from nuralNet import MSENet, SoftMaxNet
from sklearn.preprocessing import StandardScaler

# gets mnist data with test\train split
def mnist():
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    return train_X,train_y,test_X,test_y

f_name = ""

if __name__ == "__main__":
    # gat the data
    X, Y, Xt, Yt = mnist()

    # flatten the pictures
    X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
    Xt = Xt.reshape(Xt.shape[0],Xt.shape[1]*Xt.shape[2])

    # scale the data to make learning easier
    std_scale = StandardScaler()
    X = std_scale.fit_transform(X)
    Xt = std_scale.transform(Xt)

    # create the network
    nn = MSENet([28*28,200,100,10],lr=1e-2,alpha=0)

    # load previous save
    if isfile(f_name):
        nn.load(f_name)

    # fit the model
    nn.fit(X,Y,epochs=100,batch_size=8)
    
    # save best weights
    #nn.save(f_name)

    # score it
    print("error rate",nn.score(Xt,Yt))