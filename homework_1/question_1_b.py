from deepul.hw1_helper import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data


class Distribution(nn.Module):
    def __init__(self, dimension, num_logistics=4):
        super(Distribution, self).__init__()
        self.dimension = dimension
        self.num_logistics = num_logistics
        self.logits = nn.Parameter(torch.zeros(num_logistics, dtype=torch.float))
        self.means = nn.Parameter(torch.linspace(0, dimension, num_logistics, dtype=torch.float))
        self.inv_scales = nn.Parameter(torch.exp(-torch.randn(num_logistics, dtype=torch.float)))

    def get_probability(self, x):
        infinity = 10000000.
        x = x.unsqueeze(1)

        start_integration = torch.where(x > 0.001, torch.tensor(-0.5), torch.tensor(-infinity))
        end_integration = torch.where(x < self.dimension - 1.001, torch.tensor(0.5), torch.tensor(infinity))

        cdf_plus = torch.sigmoid((x + end_integration - self.means) * self.inv_scales)
        cdf_minus = torch.sigmoid((x + start_integration - self.means) * self.inv_scales)

        x_log_prob = torch.log(torch.clamp(cdf_plus - cdf_minus, min=1e-12))
        pi_log_prob = torch.log_softmax(self.logits, dim=0)
        log_prob = x_log_prob + pi_log_prob

        sum = torch.sum(torch.exp(log_prob), dim=1)
        return sum

    def get_loss(self, x):
        return - torch.mean(torch.log(self.get_probability(x)))

    def get_distribution(self):
        with torch.no_grad():
            x = torch.FloatTensor(np.arange(self.dimension))
            distribution = self.get_probability(x)
        return distribution.detach().cpu().numpy()


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


def q1_b(train_data, test_data, dimension, dataset_id):

    histogram = Distribution(dimension)
    optimizer = optim.Adam(histogram.parameters(), lr=0.1)

    train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(test_data, batch_size=128)

    num_epochs = 10

    train_losses, test_losses = train(histogram, optimizer, train_loader, test_loader, num_epochs)
    return np.array(train_losses), np.array(test_losses), histogram.get_distribution()


q1_save_results(1, 'b', q1_b)
q1_save_results(2, 'b', q1_b)