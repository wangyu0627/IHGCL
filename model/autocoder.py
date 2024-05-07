from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch

class Autoencoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, enc_num_layer, dec_num_layer,
                 mask_rate, remask_rate, num_remasking):
        super(Autoencoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        # encoder
        for i in range(enc_num_layer):
            self.encoder.append(GraphConv(in_dim, hidden_dim, bias=False, weight=False,
                                          allow_zero_in_degree=True))

        # decoder
        for i in range(dec_num_layer):
            self.decoder.append(GraphConv(in_dim, hidden_dim, bias=False, weight=False,
                                          allow_zero_in_degree=True))

        # random_mask
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.num_remasking = num_remasking

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        self.encoder_to_decoder = nn.Linear(in_dim, hidden_dim, bias=False)
        self.decoder_to_contrastive = nn.Linear(in_dim, hidden_dim, bias=False)

        self.reset_parameters_for_token()

    def reset_parameters_for_token(self):
        nn.init.xavier_normal_(self.enc_mask_token)
        nn.init.xavier_normal_(self.dec_mask_token)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)

    def forward(self, g, x, drop_g1=None, drop_g2=None):
        # mask
        pre_use_g, mask_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self.mask_rate)
        use_g = drop_g1 if drop_g1 is not None else g
        # multi-layer encoder
        Encode = []
        for i, layer in enumerate(self.encoder):
            enc_rep = layer(use_g, mask_x)
            Encode.append(enc_rep)

        Es = torch.stack(Encode, dim=1)  # (N, M, D * K)
        Es = torch.mean(Es, dim=1)
        # encode_to_decode
        origin_rep = self.encoder_to_decoder(Es)
        # decode
        Decode = []
        for i in range(self.num_remasking):
            # remask
            rep = origin_rep.clone()
            rep, remask_nodes, rekeep_nodes = self.random_remask(pre_use_g, rep, self.remask_rate)
            # multi-layer decoder
            for i, layer in enumerate(self.encoder):
                recon = layer(pre_use_g, rep).flatten(1)
                Decode.append(recon)

            Ds = torch.stack(Decode, dim=1)  # (N, M, D * K)
            Ds = torch.mean(Ds, dim=1)

        return  self.decoder_to_contrastive(Ds)

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        out_x = x.clone()
        token_nodes = mask_nodes
        out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def random_remask(self, g, rep, remask_rate=0.5):

        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=rep.device)
        num_remask_nodes = int(remask_rate * num_nodes)
        remask_nodes = perm[: num_remask_nodes]
        rekeep_nodes = perm[num_remask_nodes:]

        rep = rep.clone()
        rep[remask_nodes] = 0
        rep[remask_nodes] += self.dec_mask_token

        return rep, remask_nodes, rekeep_nodes
