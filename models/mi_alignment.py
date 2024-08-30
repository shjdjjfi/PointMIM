import torch
import torch.nn as nn

class MutualInformationAlignment(nn.Module):
    def __init__(self, feature_dim=1024):
        super(MutualInformationAlignment, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def mutual_information(self, x, y):
        p_joint = torch.bmm(x.unsqueeze(2), y.unsqueeze(1))
        p_joint = p_joint / p_joint.sum(dim=-1, keepdim=True)
        p_x = x.mean(dim=0, keepdim=True)
        p_y = y.mean(dim=0, keepdim=True)
        p_prod = torch.bmm(p_x.unsqueeze(2), p_y.unsqueeze(1))
        mi = (p_joint * (torch.log(p_joint + 1e-9) - torch.log(p_prod + 1e-9))).sum()
        return mi

    def forward(self, x, y):
        # x 和 y 是输入的特征张量
        x = self.fc(x)
        y = self.fc(y)
        mi = self.mutual_information(x, y)
        return mi
