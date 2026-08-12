"""
SCM-UNet: Spatial-Channel Mamba UNet for medical image segmentation.

Reimplementation from:
  Yan et al., "SCM-UNet: Spatial-channel Mamba UNet for medical image
  segmentation", Digital Signal Processing 168 (2026) 105550.

This is an independent, from-scratch reimplementation based on the
architecture description in the paper (Sections 3.1-3.4, Fig. 2-6,
Algorithm 1). It does not use the authors' original code.

Notable implementation choice: the selective-scan (S6) core of SS2D is
implemented as a fully-vectorized parallel scan in pure PyTorch (no
custom CUDA kernel / mamba_ssm dependency required), so the code runs
anywhere PyTorch runs. See `selective_scan` below.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Selective scan (S6) core
# --------------------------------------------------------------------------

def selective_scan(x, delta, A, B, C, D=None):
    """Vectorized (parallel, log-space) selective scan.

    Implements the discretized linear state-space recurrence:
        h_t = Abar_t * h_{t-1} + Bbar_t * x_t
        y_t = C_t . h_t (+ D * x_t)
    with a diagonal state matrix A (per-channel), computed for the whole
    sequence at once via cumulative sums in log-space instead of a
    sequential Python loop.

    Args:
        x:     (B, L, D_in)              input sequence
        delta: (B, L, D_in)               time-step (already softplus'd, >0)
        A:     (D_in, N)                  continuous state matrix (negative)
        B:     (B, L, N)                  input projection (data-dependent)
        C:     (B, L, N)                  output projection (data-dependent)
        D:     (D_in,) or None            skip connection
    Returns:
        y: (B, L, D_in)
    """
    Bsz, L, Din = x.shape
    N = A.shape[-1]

    # Discretize: Abar = exp(delta * A)   (B, L, Din, N)
    deltaA = torch.exp(delta.unsqueeze(-1) * A)  # A is negative -> deltaA in (0,1)
    deltaB_x = delta.unsqueeze(-1) * B.unsqueeze(2) * x.unsqueeze(-1)  # (B,L,Din,N)

    # Parallel scan via log-space cumulative product:
    #   P_t = prod_{s<=t} Abar_s
    #   h_t = P_t * cumsum_{s<=t}( Bbar_s x_s / P_s )
    eps = 1e-12
    log_deltaA = torch.log(deltaA.clamp(min=eps))
    cum_log_A = torch.cumsum(log_deltaA, dim=1)  # (B,L,Din,N)
    cum_log_A = cum_log_A.clamp(min=-30.0, max=0.0)
    P = torch.exp(cum_log_A)

    safe_P = P.clamp(min=eps)
    scaled = deltaB_x / safe_P
    cum_scaled = torch.cumsum(scaled, dim=1)
    h = P * cum_scaled  # (B, L, Din, N)

    y = torch.einsum('bldn,bln->bld', h, C)
    if D is not None:
        y = y + x * D
    return y


class S6Block(nn.Module):
    """Single-direction selective SSM (the 'S6 structural unit' of the paper),
    operating on a flattened 1D sequence of length L with D_in channels.
    """

    def __init__(self, d_inner, d_state=16, dt_rank=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, d_inner // 16)

        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # A: (d_inner, d_state), stored in log-space, kept negative via -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            inv_softplus = dt + torch.log(-torch.expm1(-dt))
            self.dt_proj.bias.copy_(inv_softplus)

    def forward(self, x):
        # x: (B, L, d_inner)
        A = -torch.exp(self.A_log)  # (d_inner, d_state), negative
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt, Bp, Cp = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_inner)
        y = selective_scan(x, dt, A, Bp, Cp, self.D)
        return y


def _scan_order(H, W, orientation):
    """Return a permutation (flat index LongTensor of length H*W) that
    reorders a row-major (H, W) grid according to one of 4 scan directions,
    matching Fig. 4 of the paper (cross-scan in 4 orientations). We use the
    standard VMamba-style 4-way scan: row-major, column-major, and their
    reverses, realizing the four "corner-to-corner" orientations described
    in the paper.
    """
    idx = torch.arange(H * W).view(H, W)
    if orientation == 0:  # top-left -> bottom-right, row-major
        order = idx.flatten()
    elif orientation == 1:  # bottom-right -> top-left (reverse of 0)
        order = idx.flatten().flip(0)
    elif orientation == 2:  # top-right -> bottom-left, column-major
        order = idx.t().flatten()
    else:  # bottom-left -> top-right (reverse of 2)
        order = idx.t().flatten().flip(0)
    return order


class SS2D(nn.Module):
    """2D Selective Scan module (Algorithm 1 in the paper).

    Input/Output: (B, H, W, C)
    """

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.dwconvs = nn.ModuleList([
            nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv,
                      padding=d_conv // 2, groups=self.d_inner, bias=True)
            for _ in range(4)
        ])
        self.s6_blocks = nn.ModuleList([
            S6Block(self.d_inner, d_state=d_state) for _ in range(4)
        ])

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._cached_orders = {}

    def _get_orders(self, H, W, device):
        key = (H, W, device)
        if key not in self._cached_orders:
            orders = [_scan_order(H, W, o).to(device) for o in range(4)]
            inv_orders = [torch.argsort(o) for o in orders]
            self._cached_orders[key] = (orders, inv_orders)
        return self._cached_orders[key]

    def forward(self, u):
        # u: (B, H, W, C)
        B, H, W, C = u.shape
        xz = self.in_proj(u)  # (B,H,W,2*d_inner)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, d_inner, H, W)

        orders, inv_orders = self._get_orders(H, W, u.device)
        L = H * W
        y_sum = 0
        for o in range(4):
            xo = F.silu(self.dwconvs[o](x))  # (B, d_inner, H, W)
            seq = xo.flatten(2).transpose(1, 2)  # (B, L, d_inner) row-major
            seq = seq[:, orders[o], :]  # apply this direction's scan order
            yo = self.s6_blocks[o](seq)  # (B, L, d_inner)
            yo = yo[:, inv_orders[o], :]  # back to row-major order
            y_sum = y_sum + yo

        y = self.out_norm(y_sum)
        y = y * F.silu(z.reshape(B, L, -1))
        y = self.out_proj(y)
        y = self.dropout(y)
        return y.view(B, H, W, self.d_model)


# --------------------------------------------------------------------------
# VSSLayer (Fig. 3)
# --------------------------------------------------------------------------

class VSSLayer(nn.Module):
    """Visual State Space layer: dual-branch fusion of an SS2D branch and a
    depthwise-conv channel branch, gated by multiplication, with a residual
    connection, matching Fig. 3.
    """

    def __init__(self, dim, d_state=16, expand=2.0, drop=0.0):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)

        hidden = int(dim * expand)
        self.branch1_lin = nn.Linear(dim, hidden)
        self.branch1_dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.ss2d = SS2D(hidden, d_state=d_state, expand=1.0, dropout=drop)
        self.branch1_norm = nn.LayerNorm(hidden)

        self.branch2_lin = nn.Linear(dim, hidden)

        self.fuse_lin = nn.Linear(hidden, dim)
        self.final_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        # x: (B, H, W, C)
        shortcut = x
        xn = self.norm_in(x)

        # branch 1: linear -> dwconv+silu -> SS2D -> LN
        b1 = self.branch1_lin(xn)
        b1_ = b1.permute(0, 3, 1, 2).contiguous()
        b1_ = F.silu(self.branch1_dwconv(b1_))
        b1_ = b1_.permute(0, 2, 3, 1).contiguous()
        b1_ = self.ss2d(b1_)
        b1_ = self.branch1_norm(b1_)

        # branch 2: linear -> silu (gate)
        b2 = F.silu(self.branch2_lin(xn))

        fused = b1_ * b2
        fused = self.fuse_lin(fused)
        fused = self.drop(fused)

        out = shortcut + fused
        out = self.final_norm(out)
        return out


# --------------------------------------------------------------------------
# Down/Up sampling
# --------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """ImageTokenizer: image -> initial token grid."""

    def __init__(self, in_chans=3, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                               stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # (B,C,H,W)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        x = self.norm(x)
        return x


class DownsamplingLayer(nn.Module):
    """2x spatial downsampling with channel doubling."""

    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        # x: (B,H,W,C) -> (B,H/2,W/2,2C)
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class UpsamplingLayer(nn.Module):
    """2x spatial upsampling with channel halving."""

    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        # x: (B,H,W,C) -> (B,2H,2W,C/2)
        B, H, W, C = x.shape
        x = self.expand(x)  # (B,H,W,2C)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        return x


class FinalUpsampling(nn.Module):
    """Restores full input resolution and predicts the segmentation mask."""

    def __init__(self, dim, out_chans=1, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, dim * patch_size * patch_size, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Conv2d(dim, out_chans, kernel_size=1)

    def forward(self, x):
        B, H, W, C = x.shape
        p = self.patch_size
        x = self.expand(x)
        x = x.view(B, H, W, p, p, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * p, W * p, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        x = self.head(x)
        return x


# --------------------------------------------------------------------------
# SC-Att Bridge (Fig. 5): Spatial Attention Bridge (SAB) + Channel
# Attention Bridge (CAB)
# --------------------------------------------------------------------------

class SpatialAttentionBridge(nn.Module):
    """Shared spatial attention applied identically to every stage's
    feature map (Eq. 9-12)."""

    def __init__(self, kernel_size=7, dilation=3):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, dilation=dilation,
                               padding=padding, bias=True)

    def forward(self, feats):
        # feats: list of (B,C_i,H_i,W_i) tensors
        out = []
        for f in feats:
            avg = torch.mean(f, dim=1, keepdim=True)
            mx, _ = torch.max(f, dim=1, keepdim=True)
            a = torch.cat([avg, mx], dim=1)  # (B,2,H,W)
            Ms = torch.sigmoid(self.conv(a))  # (B,1,H,W), shared conv weights
            out.append(f * Ms)
        return out


class ChannelAttentionBridge(nn.Module):
    """Cross-stage channel attention (Eq. 13-16): pools every stage's
    feature map with GAP, concatenates the descriptors, mixes them with a
    1D conv across stages, then produces per-stage channel gates.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        total = sum(channels)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.fcs = nn.ModuleList([nn.Linear(total, c) for c in channels])

    def forward(self, feats):
        zs = [F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats]  # (B,c_i)
        z = torch.cat(zs, dim=1)  # (B, sum_c)
        z = self.conv1d(z.unsqueeze(1)).squeeze(1)  # (B, sum_c)
        out = []
        for i, f in enumerate(feats):
            a = torch.sigmoid(self.fcs[i](z))  # (B, c_i)
            a = a.unsqueeze(-1).unsqueeze(-1)
            out.append(f + a * f)
        return out


class SCAttBridge(nn.Module):
    """Full SC-Att Bridge: SAB followed by CAB, operating jointly across
    all encoder-stage skip features (Section 3.3)."""

    def __init__(self, channels):
        super().__init__()
        self.sab = SpatialAttentionBridge()
        self.cab = ChannelAttentionBridge(channels)

    def forward(self, feats):
        # feats: list of (B,C_i,H_i,W_i), one per encoder stage
        feats = self.sab(feats)
        feats = self.cab(feats)
        return feats


# --------------------------------------------------------------------------
# KANLinear (Section 3.4 / Fig. 6)
# --------------------------------------------------------------------------

class KANLinear(nn.Module):
    """A single Kolmogorov-Arnold Network linear layer using learnable
    B-spline activations on each input-output connection, following
    Liu et al. (2024) and as used at the SCM-UNet bottleneck.
    """

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 grid_range=(-1, 1)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer('grid', grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(grid_size + 1, in_features, out_features) - 0.5)
                * scale_noise / grid_size
            )
            sample_x = self.grid.T[spline_order:-spline_order]  # (grid_size+1, in_features)
            coeff = self._curve2coeff(sample_x, noise)
            self.spline_weight.data.copy_(coeff * scale_spline)

    def _b_splines(self, x):
        # x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)] + 1e-8)
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k] + 1e-8)
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def _curve2coeff(self, x, y):
        # x: (points, in_features), y: (points, in_features, out_features)
        A = self._b_splines(x).transpose(0, 1)  # (in_features, points, coeff)
        B = y.transpose(0, 1)  # (in_features, points, out_features)
        sol = torch.linalg.lstsq(A, B).solution  # (in_features, coeff, out_features)
        return sol.permute(2, 0, 1)  # -> (out_features, in_features, coeff)

    def forward(self, x):
        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)
        base_out = F.linear(F.silu(x), self.base_weight)
        spline_basis = self._b_splines(x)  # (batch, in_features, coeff)
        spline_out = F.linear(
            spline_basis.reshape(x.size(0), -1),
            self.spline_weight.reshape(self.out_features, -1),
        )
        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)


class KANBottleneck(nn.Module):
    """Bottleneck connection: W' = LN(Z + Conv(Phi(Z))), Eq. 17."""

    def __init__(self, dim):
        super().__init__()
        self.kan = KANLinear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B,H,W,C)
        z = self.kan(x)  # (B,H,W,C)
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.conv(z)
        z = z.permute(0, 2, 3, 1).contiguous()
        out = self.norm(x + z)
        return out


# --------------------------------------------------------------------------
# Full SCM-UNet
# --------------------------------------------------------------------------

class SCM_UNet(nn.Module):
    """Encoder: 4 VSSLayer stages, downsampling 2x after stages 0,1,2 (stage 3
    stays at the deepest resolution and feeds straight into the KAN
    bottleneck) -- matching Fig. 2, where only 3 SC-Att-Bridge arrows connect
    encoder stages 1-3 to the decoder, while VSSLayer4's output goes directly
    into the KANLinear bottleneck.

    Decoder: 3 (upsample + skip-fusion + VSSLayer) stages mirroring the 3
    skip connections, followed by one extra refinement VSSLayer (matching the
    4th 'VSSLayer_up' block in Fig. 2) and a FinalUpsampling module that
    restores the original input resolution and predicts the mask.
    """

    def __init__(self, in_chans=3, num_classes=1, base_dim=64, depths=(1, 1, 1, 1),
                 d_state=16, patch_size=4, drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, base_dim, patch_size)

        dims = [base_dim * (2 ** i) for i in range(4)]  # e.g. 64,128,256,512

        # ---------------- Encoder ----------------
        self.enc_stages = nn.ModuleList()
        self.enc_down = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(*[
                VSSLayer(dims[i], d_state=d_state, drop=drop) for _ in range(depths[i])
            ])
            self.enc_stages.append(stage)
            if i < 3:
                self.enc_down.append(DownsamplingLayer(dims[i]))
            else:
                self.enc_down.append(None)

        # ---------------- Bottleneck ----------------
        self.bottleneck = KANBottleneck(dims[3])

        # ---------------- Skip fusion (3 skips: stages 0,1,2) ----------------
        self.sc_bridge = SCAttBridge(dims[:3])

        # ---------------- Decoder ----------------
        # 3 upsample+skip stages: d3->d2 (+skip2), d2->d1 (+skip1), d1->d0 (+skip0)
        self.dec_up = nn.ModuleList([
            UpsamplingLayer(dims[3]),
            UpsamplingLayer(dims[2]),
            UpsamplingLayer(dims[1]),
        ])
        self.dec_stages = nn.ModuleList([
            nn.Sequential(*[VSSLayer(dims[2], d_state=d_state, drop=drop)
                             for _ in range(depths[2])]),
            nn.Sequential(*[VSSLayer(dims[1], d_state=d_state, drop=drop)
                             for _ in range(depths[1])]),
            nn.Sequential(*[VSSLayer(dims[0], d_state=d_state, drop=drop)
                             for _ in range(depths[0])]),
        ])
        # extra refinement stage (no upsample / no skip) -> 4th VSSLayer_up
        self.dec_refine = nn.Sequential(*[
            VSSLayer(dims[0], d_state=d_state, drop=drop) for _ in range(depths[0])
        ])

        self.final_up = FinalUpsampling(dims[0], out_chans=num_classes,
                                         patch_size=patch_size)

    def forward(self, x):
        x = self.patch_embed(x)  # (B,H0,W0,dim0)

        skips = []
        cur = x
        for i in range(4):
            cur = self.enc_stages[i](cur)
            if i < 3:
                skips.append(cur)  # (B,Hi,Wi,Ci) for stages 0,1,2
                cur = self.enc_down[i](cur)
            # stage 3 output (i == 3) is not downsampled -> feeds bottleneck

        # bottleneck (operates at stage-3 resolution/channels)
        cur = self.bottleneck(cur)

        # SC-Att bridge fusion over the 3 encoder skips (channel-first)
        skips_cf = [s.permute(0, 3, 1, 2).contiguous() for s in skips]
        skips_cf = self.sc_bridge(skips_cf)
        skips = [s.permute(0, 2, 3, 1).contiguous() for s in skips_cf]

        dec = cur
        for i in range(3):
            dec = self.dec_up[i](dec)
            skip = skips[2 - i]  # stage2, stage1, stage0 in that order
            dec = dec + skip
            dec = self.dec_stages[i](dec)

        dec = self.dec_refine(dec)
        out = self.final_up(dec)
        return out


def build_mamba_unet(num_classes=1, base_dim=64, patch_size=4, d_state=16):
    """Build a Mamba-UNet model from scratch."""
    return SCM_UNet(in_chans=3, num_classes=num_classes, base_dim=base_dim,
                     patch_size=patch_size, d_state=d_state)


def build_scm_unet(num_classes=1, base_dim=64, patch_size=4, d_state=16):
    """Backward-compatible alias for SCM-UNet."""
    return build_mamba_unet(num_classes=num_classes, base_dim=base_dim,
                            patch_size=patch_size, d_state=d_state)


if __name__ == '__main__':
    model = build_mamba_unet(num_classes=1, base_dim=32, patch_size=4, d_state=8)
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print('output shape:', y.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params/1e6:.2f}M')
