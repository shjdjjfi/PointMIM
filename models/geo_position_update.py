import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn

class GeometricPositionUpdate(nn.Module):
    def __init__(self, k=8, feature_dim=1024):
        super(GeometricPositionUpdate, self).__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x, pos):
        idx = knn(pos, pos, self.k)
        x_knn = x[idx] 
        x_cat = torch.cat([x.unsqueeze(2).repeat(1, 1, self.k, 1), x_knn], dim=-1) 
        x_updated = self.mlp(x_cat) 
        x_updated = torch.max(x_updated, dim=2)[0] 
        return x_updated
