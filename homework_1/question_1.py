from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

class Histogram(nn.Module):
    def __init__(self, dimension):
        super(Histogram, self).__init__()
        self.dimension = dimension
        self.theta = nn.Parameter(torch.zeros(dimension, dtype=torch.float))

    def get_loss(self, x):
        return F.cross_entropy(self.theta.unsqueeze(0).repeat(x.shape[0], 1), x.long())

    def get_distribution(self):
        return F.softmax(self.theta.detach()).numpy()

def q1_a(train_data, test_data, dimension, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
               used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """
    train_losses = []
    test_losses = []

    histogram = Histogram(dimension)
    optimizer = optim.SGD(histogram.parameters(), lr=0.1)

    test_loader = data.DataLoader(test_data, batch_size=256, shuffle=True)
    train_loader = data.DataLoader(train_data, batch_size=256, shuffle=True)

    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_train in train_loader:
            loss = histogram.get_loss(batch_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        test_loss = sum(histogram.get_loss(batch_test) for batch_test in test_loader).item()
        test_losses.append(test_loss)
        print(test_loss)

    return np.array(train_losses), np.array(test_losses), histogram.get_distribution()

q1_save_results(1, 'a', q1_a)