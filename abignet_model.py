import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_binary_gumbel_sigmoid(logits, tau=1.0, hard=False):
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = torch.sigmoid((logits + noise) / tau)

    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    return y


class AdjacencyGenerator(nn.Module):
    def __init__(
        self, in_dim, hidden_dim1=32, hidden_dim2=16,
        tau=1.0, hard=False, prior_pi=0.2,
        num_samples=5,
    ):
        super().__init__()

        self.tau = tau
        self.hard = hard
        self.prior_pi = prior_pi
        self.num_samples = num_samples

        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )


    def forward(self, x, return_kl=False):
        n = x.size(0)
        xi = x.unsqueeze(1).expand(n, n, -1)
        xj = x.unsqueeze(0).expand(n, n, -1)
        pair_feat = torch.cat([xi, xj], dim=-1)
        logits = self.mlp(pair_feat.reshape(n * n, -1)).reshape(n, n)
        A = 0
        # use it if A is too unstable
        #for _ in range(self.num_samples):
        #    A = A + sample_binary_gumbel_sigmoid(
        #        logits,
        #        tau=self.tau,
        #        hard=self.hard,
        #    )
        #A = A / self.num_samples
        A = sample_binary_gumbel_sigmoid(logits, tau=self.tau, hard=self.hard)
        A = (A + A.T) / 2
        A.fill_diagonal_(1.0)
        if return_kl:
            return A, self.kl_divergence(logits)
        return A


    # we did not use kl_divergence. we set the kl_divergence regularizer 0 later. 
    # however, one can use it if needed
    def kl_divergence(self, logits):
        p = torch.sigmoid(logits)
        pi = self.prior_pi

        kl = (
            p * torch.log((p + 1e-10) / (pi + 1e-10))
            + (1 - p) * torch.log((1 - p + 1e-10) / (1 - pi + 1e-10))
        )

        return kl.mean()

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))
    def forward(self, x, A):
        deg = A.sum(dim=1).clamp(min=1e-8)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        A_norm = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]
        return A_norm @ (x @ self.weight) + self.bias

class DenseGCN(nn.Module):
    def __init__(self, in_dim, out_dim=2):
        super().__init__()
        self.dropout = 0.2
        self.gcn1 = DenseGCNLayer(in_dim, 64)
        self.norm1 = nn.LayerNorm(64)
        self.gcn2 = DenseGCNLayer(64, 512)
        self.norm2 = nn.LayerNorm(512)
        self.gcn3 = DenseGCNLayer(512, 1024)
        self.norm3 = nn.LayerNorm(1024)
        # JK concatenation: 64 + 512 + 1024 = 1600
        self.classifier1 = nn.Linear(1600, 256)
        self.classifier2 = nn.Linear(256, 32)
        self.classifier3 = nn.Linear(32, out_dim)

    def forward(self, x, A):
        h1 = F.relu(self.norm1(self.gcn1(x, A)))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)
        h2 = F.relu(self.norm2(self.gcn2(h1, A)))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        h3 = F.relu(self.norm3(self.gcn3(h2, A)))
        h3 = F.dropout(h3, p=self.dropout, training=self.training)
        # Jumping Knowledge aggregation from all 3 GCN layers
        g1 = h1.mean(dim=0, keepdim=True)
        g2 = h2.mean(dim=0, keepdim=True)
        g3 = h3.mean(dim=0, keepdim=True)
        graph_emb = torch.cat([g1, g2, g3], dim=1)
        out = F.relu(self.classifier1(graph_emb))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.classifier2(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.classifier3(out)