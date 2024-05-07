import torch
import torch.nn as nn
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli, LogitRelaxedBernoulli
import dgl
class ViewLearner(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ViewLearner, self).__init__()
        self.mlp_edge_model = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1)
        )

    def build_prob_neighbourhood(self, reg, edge_weight, temperature):
        attention = torch.clamp(edge_weight, 0.01, 0.99)

        relaxed_bernoulli = RelaxedBernoulli(temperature=torch.tensor([temperature]).to(attention.device),
                                             probs=attention)

        weighted_adjacency_matrix = relaxed_bernoulli.rsample()
        eps = 0.0
        mask = (weighted_adjacency_matrix > eps).detach().float()
        weighted_adjacency_matrix = weighted_adjacency_matrix * mask + 0.0 * (1 - mask)

        return weighted_adjacency_matrix

    def forward(self, g, node_emb):
        src, dst = g.edges()
        emb_src = node_emb[src]
        emb_dst = node_emb[dst]

        edge_emb = torch.cat([emb_src, emb_dst], dim=1)
        edge_logits = self.mlp_edge_model(edge_emb)
        temperature = 1.0  # The temperature parameter can be adjusted if needed
        bias = 0.0001  # Small bias to avoid numerical issues

        # Reparameterization trick
        eps = (bias - (1 - bias)) * torch.rand(edge_logits.size(), device=edge_logits.device) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = (gate_inputs + edge_logits) / temperature
        edge_weight = torch.sigmoid(gate_inputs).squeeze()
        weighted_adjacency_matrix = self.build_prob_neighbourhood(g, edge_weight, 0.9)
        mask = (weighted_adjacency_matrix != 0)
        filtered_src = src[mask]
        filtered_dst = dst[mask]
        adj = dgl.graph((filtered_src, filtered_dst), num_nodes=g.num_nodes())
        return adj

