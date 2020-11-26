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
        return F.softmax(self.theta.detach(), dim=0).numpy()


def train_model(model, optimizer, train_loader):
    losses = []
    for batch_train in train_loader:
        loss = model.get_loss(batch_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


def eval_model(model, test_loader):
    total_loss = 0
    with torch.no_grad():
        for batch_test in test_loader:
            total_loss += model.get_loss(batch_test).item()*len(batch_test)
    return total_loss/len(test_loader.dataset)


def train(model, optimizer, train_loader, test_loader, epochs):
    train_losses = []
    test_losses = [eval_model(model, test_loader)]
    for epoch in range(epochs):
        train_losses.extend(train_model(model, optimizer, train_loader))
        test_losses.append(eval_model(model, test_loader))
    return train_losses, test_losses


def q1_a(train_data, test_data, dimension, dataset_id):

    histogram = Histogram(dimension)
    optimizer = optim.Adam(histogram.parameters(), lr=0.1)

    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)

    num_epochs = 10

    train_losses, test_losses = train(histogram, optimizer, train_loader, test_loader, num_epochs)
    return np.array(train_losses), np.array(test_losses), histogram.get_distribution()


q1_save_results(1, 'a', q1_a)
q1_save_results(2, 'a', q1_a)