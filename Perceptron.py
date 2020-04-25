import torch
import torch.nn as nn


class Perceptron(nn.Module):
    """ a simple perceptron based classifier """

    def __init__(self, num_features):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        super(Perceptron, self).__init__()
        #self.fc1 = nn.Linear(in_features=num_features, out_features=1)
        self.fc1 = nn.Linear(in_features=num_features, out_features=2)

    def forward(self, x_in, apply_sigmoid=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, num_features)
            apply_sigmoid (bool): a flag for the sigmoid activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        x_in = x_in.float()
        #y_out = self.fc1(x_in).squeeze()
        y_out = self.fc1(x_in)
        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out