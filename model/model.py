import utility.losses
import utility.tools
import utility.trainer
from .ViewLearner import ViewLearner
from .autocoder import Autoencoder
from dgl.nn.pytorch import GraphConv
import torch
import torch.nn.functional as F
import math
from torch import nn
import utility.losses
import utility.tools
import utility.trainer

class GCRec(nn.Module):
    def __init__(self, config, dataset, user_g, item_g, device):
        super(GCRec, self).__init__()
        self.config = config
        self.dataset = dataset
        self.device = device
        self.reg_lambda = float(self.config.reg_lambda)
        self.ssl_lambda = float(self.config.ssl_lambda)
        self.ib_lambda = float(self.config.ib_lambda)
        self.intra_lambda = float(self.config.intra_lambda)
        self.temperature = float(self.config.temperature)
        self.view_learner = ViewLearner(input_dim=self.config.dim, output_dim=self.config.dim)
        self.IB_size = self.config.IB_size
        self.IB_2_size = int(self.config.IB_size/2)
        self.user_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_users, embedding_dim=int(self.config.dim))
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.dataset.num_items, embedding_dim=int(self.config.dim))

        # no pretrain
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1)

        self.Graph = self.dataset.sparse_adjacency_matrix()  # sparse matrix
        self.Graph = utility.tools.convert_sp_mat_to_sp_tensor(self.Graph)  # sparse tensor
        self.Graph = self.Graph.coalesce().to(self.device)  # Sort the edge index and remove redundancy
        self.activation = nn.Sigmoid()
        # hete_information
        self.uu_graph = user_g
        self.ii_graph = item_g
        self.user_autoencoder = nn.ModuleList()
        self.user_compressor = nn.Linear(self.config.dim, self.config.IB_size)
        self.item_compressor = nn.Linear(self.config.dim, self.config.IB_size)
        for i in range(len(user_g)):
            self.user_autoencoder.append(Autoencoder(in_dim=config.in_size, hidden_dim=config.out_size,
                                            enc_num_layer=config.enc_num_layer, dec_num_layer=config.dec_num_layer,
                                            mask_rate=config.mask_rate, remask_rate=config.remask_rate, num_remasking=config.num_remasking))
            # ablation
            # self.user_autoencoder.append(GraphConv(config.in_size, config.out_size, bias=False, weight=False,
            #                               allow_zero_in_degree=True))
        self.item_autoencoder = nn.ModuleList()
        for i in range(len(item_g)):
            self.item_autoencoder.append(Autoencoder(in_dim=config.in_size, hidden_dim=config.out_size,
                                            enc_num_layer=config.enc_num_layer, dec_num_layer=config.dec_num_layer,
                                            mask_rate=config.mask_rate, remask_rate=config.remask_rate, num_remasking=config.num_remasking))
            # ablation
            # self.item_autoencoder.append(GraphConv(config.in_size, config.out_size, bias=False, weight=False,
            #                                        allow_zero_in_degree=True))
    def aggregate(self):
        # [user + item, emb_dim]
        all_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight])

        # no dropout
        embeddings = []

        for layer in range(int(self.config.GCN_layer)):
            all_embedding = torch.sparse.mm(self.Graph, all_embedding)
            embeddings.append(all_embedding)

        final_embeddings = torch.stack(embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        users_emb, items_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return users_emb, items_emb

    def forward(self, user, positive, negative, epoch=None):
        # all_user_embeddings, all_item_embeddings = self.aggregate()
        user_embeddings, item_embeddings = self.aggregate()
        # hete_emb
        hete_user_embedding = []
        hete_item_embedding = []
        for i in range(len(self.uu_graph)):
            # reparameterization
            self.uu_graph[i] = self.view_learner(self.uu_graph[i], self.user_embedding.weight)
            hete_user_embedding.append(self.user_autoencoder[i](self.uu_graph[i], self.user_embedding.weight).flatten(1))
        for i in range(len(self.ii_graph)):
            # reparameterization
            self.ii_graph[i] = self.view_learner(self.ii_graph[i], self.item_embedding.weight)
            hete_item_embedding.append(self.item_autoencoder[i](self.ii_graph[i], self.item_embedding.weight).flatten(1))

        # ib_loss
        user_node_embs = torch.mean(torch.stack(hete_user_embedding, 0), dim=0)
        item_node_embs = torch.mean(torch.stack(hete_item_embedding, 0), dim=0)
        user_node_embs = self.user_compressor(user_node_embs)
        item_node_embs = self.item_compressor(item_node_embs)
        # KL divergence
        user_mu = user_node_embs[:, :self.IB_2_size]
        user_std = F.softplus(user_node_embs[:, self.IB_2_size:] - self.IB_2_size, beta=1)

        item_mu = item_node_embs[:, :self.IB_2_size]
        item_std = F.softplus(item_node_embs[:, self.IB_2_size:] - self.IB_2_size, beta=1)

        user_kl_loss = -0.5 * (1 + 2 * user_std.log() - user_mu.pow(2) - user_std.pow(2)).sum(1).mean().div(math.log(2))
        item_kl_loss = -0.5 * (1 + 2 * item_std.log() - item_mu.pow(2) - item_std.pow(2)).sum(1).mean().div(math.log(2))

        ib_loss = self.ib_lambda * (user_kl_loss + item_kl_loss)


        user_embedding_1, user_embedding_2 = (user_embeddings + hete_user_embedding[0],
                                              user_embeddings + hete_user_embedding[1])
        item_embedding_1, item_embedding_2 = (item_embeddings + hete_item_embedding[0],
                                              item_embeddings + hete_item_embedding[1])

        all_user_embeddings, all_item_embeddings = (user_embeddings + 0.000 * user_embedding_2 + 0.000 * user_embedding_2,
                                                   item_embeddings + 0.000 * item_embedding_1 + 0.000 * item_embedding_2)

        # intra-contrast
        user_loss = []
        item_loss = []
        user_loss.append(utility.losses.get_InfoNCE_loss(hete_user_embedding[0][user.long()],
        hete_user_embedding[1][user.long()], self.temperature))

        item_loss.append(utility.losses.get_InfoNCE_loss(hete_item_embedding[0][positive.long()],
        hete_item_embedding[1][positive.long()], self.temperature))

        user_intra_loss = torch.sum(torch.stack(user_loss))
        item_intra_loss = torch.sum(torch.stack(item_loss))
        intra_loss = self.intra_lambda * (user_intra_loss + item_intra_loss)

        user_embedding = all_user_embeddings[user.long()]
        pos_embedding = all_item_embeddings[positive.long()]
        neg_embedding = all_item_embeddings[negative.long()]

        ego_user_emb = self.user_embedding(user)
        ego_pos_emb = self.item_embedding(positive)
        ego_neg_emb = self.item_embedding(negative)

        # bpr_loss
        bpr_loss = utility.losses.get_bpr_loss(user_embedding, pos_embedding, neg_embedding)

        # reg_loss
        reg_loss = utility.losses.get_reg_loss(ego_user_emb, ego_pos_emb, ego_neg_emb)
        reg_loss = self.reg_lambda * reg_loss

        # inter-contrast
        user_ssl_loss = utility.losses.get_InfoNCE_loss(user_embedding_1[user.long()],
                                                        user_embedding_2[user.long()],
                                                        self.temperature)
        item_ssl_loss = utility.losses.get_InfoNCE_loss(item_embedding_1[positive.long()],
                                                        item_embedding_2[positive.long()],
                                                        self.temperature)

        ssl_loss = self.ssl_lambda * (user_ssl_loss + item_ssl_loss)

        loss_list = [bpr_loss, reg_loss, ssl_loss, intra_loss, ib_loss]

        return loss_list

    def get_rating_for_test(self, user):
        all_user_embeddings, all_item_embeddings = self.aggregate()

        user_embeddings = all_user_embeddings[user.long()]

        rating = self.activation(torch.matmul(user_embeddings, all_item_embeddings.t()))
        return rating

    def get_embedding(self):
        all_user_embeddings, all_item_embeddings = self.aggregate()
        return all_user_embeddings, all_item_embeddings