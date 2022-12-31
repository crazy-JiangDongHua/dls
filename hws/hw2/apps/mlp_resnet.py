import sys

from numpy.random import shuffle
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    fn = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim)
    )
    return nn.Sequential(nn.Residual(fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    modules=[nn.Linear(dim, hidden_dim), nn.ReLU()]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob))
    modules.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*modules)
    ### END YOUR SOLUTION




def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    loss_func = nn.SoftmaxLoss()
    if opt is not None:
        model.train()
    else:
        model.eval()
    num_example, num_step, error_sum, loss_sum=0, 0, 0., 0.
    for X,y in dataloader:
        if opt is not None:
            opt.reset_grad()
        y_hat=model(X)
        loss=loss_func(y_hat, y)
        num_example+=y.shape[0]
        num_step+=1
        loss_sum+=loss.numpy()
        error_sum+=(y_hat.numpy().argmax(axis=1)!=y.numpy()).sum()
        if opt is not None:
            loss.backward()
            opt.step()
    return error_sum/num_example, loss_sum/num_step
    ### END YOUR SOLUTION



def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_data=ndl.data.MNISTDataset(
        data_dir+'/train-images-idx3-ubyte.gz',
        data_dir+'/train-labels-idx1-ubyte.gz'
    )
    test_data=ndl.data.MNISTDataset(
        data_dir+'/t10k-images-idx3-ubyte.gz',
        data_dir+'/t10k-labels-idx1-ubyte.gz',
    )
    train_loader=ndl.data.DataLoader(train_data, batch_size, True)
    test_loader=ndl.data.DataLoader(test_data, batch_size, False)
    model=MLPResNet(28*28, hidden_dim)
    opt=optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_error, train_loss=epoch(train_loader, model, opt)
        test_error, test_loss=epoch(test_loader, model)
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION



if __name__ == "__main__":
    train_mnist(data_dir="../data")
