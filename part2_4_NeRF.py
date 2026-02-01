import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
        self.freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs) * math.pi

    def forward(self, x):
        xb = x[..., None] * self.freq_bands.to(x.device)
        sin, cos = torch.sin(xb), torch.cos(xb)
        return torch.cat([x, sin.reshape(x.shape[0], -1), cos.reshape(x.shape[0], -1)], dim=-1)


class NeRF(nn.Module):
    def __init__(self, num_freqs_pos=10, num_freqs_dir=4, hidden_dim=256):
        super().__init__()
        self.pe_x = PositionalEncoding(num_freqs_pos)
        self.pe_d = PositionalEncoding(num_freqs_dir)

        in_x = 3 + 6 * num_freqs_pos   # 3D + sin/cos pairs
        in_d = 3 + 6 * num_freqs_dir

        self.fc_layers = nn.ModuleList([
            nn.Linear(in_x, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim + in_x, hidden_dim),  
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        ])

        self.fc_sigma = nn.Linear(hidden_dim, 1)
        self.fc_feat = nn.Linear(hidden_dim, hidden_dim)
        self.fc_color1 = nn.Linear(hidden_dim + in_d, 128)
        self.fc_color2 = nn.Linear(128, 3)

    def forward(self, x, d):
        x_enc = self.pe_x(x)
        d_enc = self.pe_d(d)

        h = x_enc
        for i in range(4):
            h = F.relu(self.fc_layers[i](h))

        # skip connection
        h = torch.cat([h, x_enc], dim=-1)

        for i in range(4, 7):
            h = F.relu(self.fc_layers[i](h))

        h = self.fc_layers[7](h)

        sigma = F.relu(self.fc_sigma(h))
        feat = self.fc_feat(h)

        h_color = torch.cat([feat, d_enc], dim=-1)
        h_color = F.relu(self.fc_color1(h_color))
        rgb = torch.sigmoid(self.fc_color2(h_color))

        return rgb, sigma


if __name__ == "__main__":
    torch.manual_seed(42)
    model = NeRF()
    x = torch.randn(1024, 3)
    d = F.normalize(torch.randn(1024, 3), dim=-1)
    rgb, sigma = model(x, d)
    print(f"rgb shape: {rgb.shape}, sigma shape: {sigma.shape}")
    print(f"rgb range: [{rgb.min():.3f}, {rgb.max():.3f}]  sigma mean: {sigma.mean():.3f}")