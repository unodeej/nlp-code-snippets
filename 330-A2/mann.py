import torch
from torch import nn, Tensor
import torch.nn.functional as F


def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = torch.nn.LSTM(num_classes + 784, hidden_dim, batch_first=True)
        self.layer2 = torch.nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        ### START CODE HERE ###
        N = self.num_classes
        K_plus_1 = self.samples_per_class
        B = input_labels.shape[0]

        # print("N = ", N)
        # print("K+1 = ", K_plus_1)

        input_tensor_all_batches = []
        labels = input_labels.clone()

        for i, b in enumerate(labels):
            input_tensor = []
            for j, s in enumerate(b):
                for k, img in enumerate(s):
                    if (j == K_plus_1 - 1):
                        labels[i,j,k] = torch.zeros(N)
                        input_tensor.append(torch.cat((input_images[i,j,k], labels[i,j,k]), dim=0))
                    else:
                        input_tensor.append(torch.cat((input_images[i,j,k], labels[i,j,k]), dim=0))
        
            input_tensor = torch.cat(tuple(input_tensor)).reshape(N*K_plus_1, 784+N)

            input_tensor_all_batches.append(input_tensor)

        input_tensor_all_batches = torch.cat(tuple(input_tensor_all_batches))
        input_tensor_all_batches = torch.reshape(input_tensor_all_batches, (B, N*K_plus_1, 784+N))

        input_tensor_all_batches = input_tensor_all_batches.type(torch.FloatTensor)

        # print(input_tensor_all_batches.shape)

        output, (h_n, c_n) = self.layer1(input_tensor_all_batches)
        # print(output.shape)
        output, (h_n, c_n) = self.layer2(output)
        # print(output.shape)

        output = torch.reshape(output, (B, K_plus_1, N, N))
        # print(output.shape)

        return output
        ### END CODE HERE ###

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        #############################

        loss = None

        ### START CODE HERE ###
        N = self.num_classes
        K_plus_1 = self.samples_per_class

        # print("N = ", N)
        # print("K+1 = ", K_plus_1)

        # print(preds.shape)
        # print(labels.shape)

        preds = preds[:,K_plus_1 - 1,:,:]
        labels = labels[:,K_plus_1 - 1,:,:]

        preds = torch.swapaxes(preds, 1, 2)
        labels = torch.swapaxes(labels, 1, 2)

        # print(preds.shape)
        # print(labels.shape)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(preds, labels)
        ### END CODE HERE ###

        return loss
