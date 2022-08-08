import os
from nuralNet import MSENet, SoftMaxNet
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split

f_name = "softmax_test"

if __name__ == "__main__":
    # test of nural net on random data

    X, Y = make_classification(10000)

    X, Xt, Y, Yt = train_test_split(X,Y,test_size=0.2,random_state=0)

    nn = SoftMaxNet([20,20,20,2],lr=1e-3)

    if os.path.isfile(f_name):
        nn.load(f_name)

    nn.fit(X,Y,epochs=200,batch_size=16)

    nn.save(f_name)

    print(nn.score(Xt,Yt))