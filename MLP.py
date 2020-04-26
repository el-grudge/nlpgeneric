import torch.nn as nn
import torch.nn.functional as F

class MLPClassifier(nn.Module):
    """ A 2-layer Multilayer Perceptron for classifying surnames """

    def __init__(self, input_dim, hidden_dim, output_dim, loss_func):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1) if loss_func == 'BCEWithLogitsLoss' else nn.Linear(hidden_dim, output_dim)


    def forward(self, loss_func, x_in, apply_softmax=False):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        x_in = x_in.float()
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector).squeeze() if loss_func == 'BCEWithLogitsLoss' else self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector