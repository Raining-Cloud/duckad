from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork


class FPNBEVHead(nn.Module):
    def __init__(
        self, c3_ch=32, c4_ch=96, c5_ch=1280, mid=128, num_classes=7, bev_size=64
    ):
        super().__init__()

        self.bev_size = bev_size
        self.fpn = FeaturePyramidNetwork([c3_ch, c4_ch, c5_ch], mid)
        self.out = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, num_classes, 1),
        )

    def forward(self, c3, c4, c5):
        feats = self.fpn(OrderedDict([("c3", c3), ("c4", c4), ("c5", c5)]))
        out = F.interpolate(
            feats["c3"],
            size=(self.bev_size, self.bev_size),
            mode="bilinear",
            align_corners=False,
        )

        return self.out(out)  # B×num_classes×bev_size×bev_size


class CommandHead(nn.Module):
    def __init__(
        self, c5_ch=1280, mid=64, pooled_size=2, num_waypoints=8, num_commands=4
    ):
        super().__init__()

        self.conv = nn.Conv2d(c5_ch, mid, kernel_size=1)
        self.pool = nn.AvgPool2d(kernel_size=4, stride=3)
        self.out = nn.Linear(
            mid * pooled_size * pooled_size + 2 * num_waypoints, num_commands
        )

    def forward(self, c5, spatials):
        feat = self.pool(F.relu(self.conv(c5))).flatten(1)

        return self.out(torch.cat([feat, spatials.flatten(1)], dim=1))  # B×num_commands


class GruWaypointHead(nn.Module):
    def __init__(self, feat_dim=1280, hidden=256, num_waypoints=10, num_commands=4):
        super().__init__()

        self.num_waypoints = num_waypoints
        self.ctx_proj = nn.Linear(feat_dim, hidden)
        self.gru = nn.GRUCell(4, hidden)
        self.spatial_heads = nn.ModuleList(
            [nn.Linear(hidden, 2) for _ in range(num_commands)]
        )
        self.temporal_heads = nn.ModuleList(
            [nn.Linear(hidden, 2) for _ in range(num_commands)]
        )

    def forward(self, feat, command):
        B = feat.size(0)
        h = F.relu(self.ctx_proj(feat))  # B×hidden
        latent_pts = torch.zeros(B, 4, device=feat.device)
        spatial_pts, temporal_pts = [], []
        batch_idx = torch.arange(B, device=feat.device)

        for _ in range(self.num_waypoints):
            h = self.gru(latent_pts, h)

            spatial_cmd_heads = torch.stack(
                [head(h) for head in self.spatial_heads], dim=1
            )  # B×num_commands×2
            spatial_pt = spatial_cmd_heads[batch_idx, command]  # B×2
            spatial_pts.append(spatial_pt)

            temporal_cmd_heads = torch.stack(
                [head(h) for head in self.temporal_heads], dim=1
            )  # B×num_commands×2
            temporal_pt = temporal_cmd_heads[batch_idx, command]  # B×2
            temporal_pts.append(temporal_pt)

            latent_pts = torch.cat([spatial_pt, temporal_pt], dim=1)  # B×4

        return torch.stack(spatial_pts, dim=1), torch.stack(
            temporal_pts, dim=1
        )  # each B×N×2


class MLPWaypointHead(nn.Module):
    def __init__(
        self, feat_dim=1280, hidden=256, num_waypoints=10, num_commands=4, depth=3
    ):
        super().__init__()

        self.num_waypoints = num_waypoints
        self.num_commands = num_commands
        layers = [nn.Linear(feat_dim, hidden), nn.ReLU(inplace=True)]

        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]

        self.trunk = nn.Sequential(*layers)
        self.spatial_out = nn.Linear(hidden, num_commands * num_waypoints * 2)
        self.temporal_out = nn.Linear(hidden, num_commands * num_waypoints * 2)

    def forward(self, feat, command):
        B = feat.size(0)
        x = self.trunk(feat)
        batch_idx = torch.arange(B, device=feat.device)

        spatial = self.spatial_out(x).view(B, self.num_commands, self.num_waypoints, 2)[
            batch_idx, command
        ]
        temporal = self.temporal_out(x).view(
            B, self.num_commands, self.num_waypoints, 2
        )[batch_idx, command]

        return spatial, temporal  # each B×N×2


WAYPOINT_HEADS = {"gru": GruWaypointHead, "mlp": MLPWaypointHead}


class DuckAD(nn.Module):
    def __init__(
        self, num_waypoints=10, bev_classes=8, bev_size=64, num_commands=4, head="gru",
        pretrained=True,
    ):
        super().__init__()

        self.head_name = head
        self.num_waypoints = num_waypoints

        f = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None).features
        self.c3 = f[:7]  # B×32×28×28
        self.c4 = f[7:14]  # B×96×14×14
        self.c5 = f[14:]  # B×1280×7×7
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.bev_head = FPNBEVHead(num_classes=bev_classes, bev_size=bev_size)
        self.waypoint_head = WAYPOINT_HEADS[head](
            num_waypoints=num_waypoints, num_commands=num_commands
        )
        self.cmd_head = CommandHead(
            num_waypoints=num_waypoints, num_commands=num_commands
        )

    def forward(self, image, command):
        c3 = self.c3(image)
        c4 = self.c4(c3)
        c5 = self.c5(c4)

        bev = self.bev_head(c3, c4, c5)  # B×bev_classes×bev_size×bev_size
        feat = self.pool(c5).flatten(1)  # B×1280
        spatial_pts, temporal_pts = self.waypoint_head(feat, command)  # each B×N×2

        cmd_logits = self.cmd_head(c5.detach(), spatial_pts.detach())  # B×num_commands

        return spatial_pts, temporal_pts, bev, cmd_logits

    def param_groups(self, lr_backbone, lr_head):
        return [
            {"params": self.c3.parameters(), "lr": lr_backbone},
            {"params": self.c4.parameters(), "lr": lr_backbone},
            {"params": self.c5.parameters(), "lr": lr_backbone},
            {"params": self.bev_head.parameters(), "lr": lr_head},
            {"params": self.waypoint_head.parameters(), "lr": lr_head},
            {"params": self.cmd_head.parameters(), "lr": lr_head},
        ]


def loss_duckad(
    pred_spatial,
    pred_bev,
    gt_spatial,
    gt_bev,
    pred_cmd,
    gt_cmd,
    pred_temporal,
    gt_temporal,
    bev_weight=0.2,
):
    l_spatial = F.l1_loss(pred_spatial, gt_spatial)  # B×N×2
    l_temp = F.l1_loss(pred_temporal, gt_temporal)  # B×N×2
    l_bev = F.cross_entropy(pred_bev, gt_bev)  # B×64×64
    l_cmd = F.cross_entropy(pred_cmd, gt_cmd)  # B
    loss = l_spatial + l_temp + bev_weight * l_bev + 0.6 * l_cmd
    
    return loss, l_spatial, l_bev, l_cmd, l_temp
