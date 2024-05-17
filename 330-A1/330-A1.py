"""
Classes defining user and item latent representations in
factorization models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to zero.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MultiTaskNet(nn.Module):
    """
    Multitask factorization representation.

    Encodes both users and items as an embedding layer; the likelihood score
    for a user-item pair is given by the dot product of the item
    and user latent vectors. The numerical score is predicted using a small MLP.

    Parameters
    ----------

    num_users: int
        Number of users in the model.
    num_items: int
        Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    layer_sizes: list
        List of layer sizes to for the regression network.
    embedding_sharing: boolean, optional
        Share embedding representations for both tasks.

    """

    def __init__(self, num_users, num_items, embedding_dim=32, layer_sizes=[96, 64],
                 embedding_sharing=True):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_sharing = embedding_sharing

        if embedding_sharing:
            self.U, self.Q = self.init_shared_user_and_item_embeddings(num_users, num_items, embedding_dim)
        else:
            self.U_reg, self.Q_reg, self.U_fact, self.Q_fact = self.init_separate_user_and_item_embeddings(num_users, num_items, embedding_dim)

        self.A, self.B = self.init_user_and_item_bias(num_users, num_items)
        self.mlp_layers = self.init_mlp_layers(layer_sizes)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            A tensor of integer user IDs of shape (batch,)
        item_ids: tensor
            A tensor of integer item IDs of shape (batch,)

        Returns
        -------

        predictions: tensor
            Tensor of user-item interaction predictions of shape (batch,)
        score: tensor
            Tensor of user-item score predictions of shape (batch,)
        """
        if self.embedding_sharing:
            predictions, score = self.forward_with_embedding_sharing(user_ids, item_ids)
        else:
            predictions, score = self.forward_without_embedding_sharing(user_ids, item_ids)

        ## Make sure you return predictions and scores of shape (batch,)
        if (len(predictions.shape) > 1) or (len(score.shape) > 1):
            raise ValueError("Check your shapes!")
        
        return predictions, score
    
    def init_shared_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes shared user and item embeddings
        used in both factorization and regression tasks

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U: ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q: ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        """
        U = Q = None
        ### START CODE HERE ###
        U = ScaledEmbedding(num_users, embedding_dim)
        Q = ScaledEmbedding(num_items, embedding_dim)
        ### END CODE HERE ###
        return U, Q
    
    def init_separate_user_and_item_embeddings(self, num_users, num_items, embedding_dim):
        """
        Initializes separate user and item embeddings
        where one will be used for factorization (ie _fact) and 
        other for regression tasks (ie _reg)

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.
        embedding_dim: int, optional
            Dimensionality of the latent representations.
            

        Returns
        -------

        U_reg: first ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_reg: first ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)
        U_fact: second ScaledEmbedding layer for users
            nn.Embedding of shape (num_users, embedding_dim)
        Q_fact: second ScaledEmbedding layer for items
            nn.Embedding of shape (num_items, embedding_dim)

        Note: Order does matter here! Please declare the layers in the order
        they are returned.
        """
        U_reg = Q_reg = U_fact = Q_fact = None
        ### START CODE HERE ###
        U_reg = ScaledEmbedding(num_users, embedding_dim)
        Q_reg = ScaledEmbedding(num_items, embedding_dim)
        U_fact = ScaledEmbedding(num_users, embedding_dim)
        Q_fact = ScaledEmbedding(num_items, embedding_dim)
        ### END CODE HERE ###
        return U_reg, Q_reg, U_fact, Q_fact
    
    def init_user_and_item_bias(self, num_users, num_items):
        """
        Initializes user and item bias terms

        Parameters
        ----------

        num_users: int
            Number of users in the model.
        num_items: int
            Number of items in the model.

        Returns
        -------

        A: ZeroEmbedding layer for users
            nn.Embedding of shape (num_users, 1)
        B: ZeroEmbedding layer for items
            nn.Embedding of shape (num_items, 1)
        """
        A = B = None
        ### START CODE HERE ###
        A = ZeroEmbedding(num_users, 1)
        B = ZeroEmbedding(num_items, 1)
        ### END CODE HERE ###
        return A, B
    
    def init_mlp_layers(self, layer_sizes):
        """
        Initializes MLP layer for regression task

        Parameters
        ----------

        layer_sizes: list
            List of layer sizes to for the regression network.

        Returns
        -------

        mlp_layers: nn.ModuleList
            MLP network containing Linear and ReLU layers
        """
        mlp_layers = None
        ### START CODE HERE ###
        mlp_layers = nn.ModuleList()
        for i, size in enumerate(layer_sizes):
            if (i < len(layer_sizes) - 1):
              mlp_layers.append(nn.Linear(size, layer_sizes[i + 1]))
              mlp_layers.append(nn.ReLU())
            else:
              mlp_layers.append(nn.Linear(size, 1))
        ### END CODE HERE ###
        return mlp_layers

    def forward_with_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        ### START CODE HERE ###
        # cos = nn.CosineSimilarity()
        # for i, user_id in enumerate(user_ids):
        #     item_id = item_ids[i]

        #     u_i = self.U(torch.LongTensor([user_id]))
        #     q_j = self.Q(torch.LongTensor([item_id]))
        #     u_q = cos(u_i, q_j)
        #     a_i = self.A(torch.LongTensor([user_id]))
        #     b_j = self.B(torch.LongTensor([item_id]))
        #     u_tensors.append(u_i)
        #     q_tensors.append(q_j)
        #     u_q_tensors.append(u_q)
        #     a_tensors.append(a_i)
        #     b_tensors.append(b_j)

        U = self.U(user_ids)
        Q = self.Q(item_ids)
        U_Q = torch.sum(U * Q, 1)
        # cos_U_Q = cos(U, Q)
        A = self.A(user_ids)
        B = self.B(item_ids)

        # cos_U_Q = torch.reshape(cos_U_Q, (cos_U_Q.shape[0], ))
        A = torch.reshape(A, (A.shape[0], ))
        B = torch.reshape(B, (B.shape[0], ))

        print("U shape")
        print(U.shape)
        print("Q shape")
        print(Q.shape)
        print("U_Q shape")
        print(U_Q.shape)
        # print("cos_U_Q shape")
        # print(cos_U_Q.shape)
        print("A shape")
        print(A.shape)
        print("B shape")
        print(B.shape)

        predictions = U_Q + A + B
        score = torch.cat((U, Q, U * Q), 1)
        for layer in self.mlp_layers:
            score = layer(score)
            # print(layer)
            # print(score.shape)

        print("predictions shape")
        print(predictions.shape)
        print("score shape")
        print(score.shape)

        predictions = torch.reshape(predictions, (predictions.shape[0],))
        score = torch.reshape(score, (score.shape[0],))

        # print("predictions shape")
        # print(predictions.shape)
        # print("score shape")
        # print(score.shape)
        ### END CODE HERE ###
        return predictions, score
    
    def forward_without_embedding_sharing(self, user_ids, item_ids):
        """
        Please see forward() docstrings for reference
        """
        predictions = score = None
        ### START CODE HERE ###
        # cos = nn.CosineSimilarity()

        U_reg = self.U_reg(user_ids)
        Q_reg = self.Q_reg(item_ids)
        U_fact = self.U_fact(user_ids)
        Q_fact = self.Q_fact(item_ids)
        U_Q = torch.sum(U_fact * Q_fact, 1)
        A = self.A(user_ids)
        B = self.B(item_ids)

        A = torch.reshape(A, (A.shape[0], ))
        B = torch.reshape(B, (B.shape[0], ))

        predictions = U_Q + A + B
        score = torch.cat((U_reg, Q_reg, U_reg * Q_reg), 1)
        for layer in self.mlp_layers:
            score = layer(score)
        
        predictions = torch.reshape(predictions, (predictions.shape[0],))
        score = torch.reshape(score, (score.shape[0],))
        ### END CODE HERE ###
        return predictions, score