import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassifier(nn.Module):
    def __init__(self, initial_num_channels, num_classes, num_channels, loss_func):
        """
        Args:
            initial_num_channels (int): size of the incoming feature vector
            num_classes (int): size of the output prediction vector
            num_channels (int): constant channel size to use throughout network
        """
        super(CNNClassifier, self).__init__()

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=initial_num_channels,
                      out_channels=num_channels, kernel_size=3),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv1d(in_channels=num_channels, out_channels=num_channels,
                      kernel_size=3),
            nn.ELU()
        )
        #self.fc = nn.Linear(num_channels, num_classes)
        self.fc = nn.Linear(num_channels, 1) if loss_func == 'BCEWithLogitsLoss' else nn.Linear(num_channels, num_classes)

    def forward(self, loss_func, x_in, apply_softmax=False):
        """The forward pass of the classifier

        Args:
            x_surname (torch.Tensor): an input data tensor.
                x_surname.shape should be (batch, initial_num_channels, max_surname_length)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, num_classes)
        """
        x_in = x_in.float()
        features = self.convnet(x_in).squeeze(dim=2)
        #prediction_vector = self.fc(features)
        prediction_vector = self.fc(features).squeeze() if loss_func == 'BCEWithLogitsLoss' else self.fc(features)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector