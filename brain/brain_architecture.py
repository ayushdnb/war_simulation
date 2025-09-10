import torch.nn as nn
class TeamBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(730 * 32, 1024),
            nn.GELU(),
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 17),  # logits
        )

    def forward(self, x, action_mask=None):  # x: [B, 729, 32]
        x = x.view(x.size(0), -1)  # flatten
        logits = self.net(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1e9)
        return logits
