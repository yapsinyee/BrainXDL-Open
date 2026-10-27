"""ST-BrainXDL: spatio-temporal brain graph model.

This module contains the full ST-BrainXDL model as published in IJCAI 2024.
Scientific behaviour is preserved exactly from legacy_code/1-ST/model.py.

Changes from legacy:
- get_device, get_activation, dense_vary, GCN_vary, recurrent_cell are
  now imported from brainxdl.common rather than defined inline.
- All model logic (STBrainXDL, GraphDecoder, GraphClassifier, GRU_GCN) is
  unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import det  # noqa: F401  (used by kld_mvnormal)
from torch_geometric.nn import global_mean_pool  # noqa: F401
from torch_geometric.utils import (
    batched_negative_sampling,
    negative_sampling,
    to_dense_batch,
    unbatch,  # noqa: F401
)

from brainxdl.common.device import get_device
from brainxdl.common.layers import (
    GCN_vary,
    dense_vary,
    get_activation,
    recurrent_cell,
)
from torch_geometric.nn import TransformerConv


def stack_truncate(Sample, Mask):
    """Unpad a batch of variable-length sequences.

    Args:
        Sample: Tensor (Time x Batch x Nodes x Features).
        Mask:   Boolean tensor (Time x Batch), True = valid time step.

    Returns:
        List of length Batch, each element a tensor (valid_T x Nodes x Features).
    """
    Container = []
    Sample = Sample.permute(1, 0, 2, 3)
    Mask = Mask.permute(1, 0)
    for sample, mask in zip(Sample, Mask):
        sample = sample[mask]
        Container.append(sample)
    return Container


class GraphDecoder(nn.Module):
    """Decode latent node embeddings back into adjacency matrices."""

    def __init__(self, activation=None, dropout=0.5):
        super().__init__()
        self.act = get_activation(activation)
        self.dropout = dropout

    def forward(self, z, edge_index):
        z = F.dropout(z, self.dropout, training=self.training)
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        if self.act is not None:
            value = self.act(value)
        return value

    def forward_sb(self, z):
        """Decode a batched dense adjacency matrix from node embeddings."""
        z = F.dropout(z, self.dropout, training=self.training)
        adj = z @ torch.transpose(z, dim0=-2, dim1=-1)
        if self.act is not None:
            adj = self.act(adj)
        return adj

    def cov2corr(self, cov):
        """PyTorch covariance → correlation (defined but not called in forward)."""
        v = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        outer_v = v.unsqueeze(dim=-1) @ v.unsqueeze(dim=-2)
        return cov / outer_v

    def loss(self, input, target):
        """BCE loss for link-prediction scenario (not used in ST-BrainXDL forward)."""
        temp_sum = target.sum()
        temp_size = target.shape[0]
        weight = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = (temp_size * temp_size) / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(
            input=input, target=target, pos_weight=weight, reduction="none"
        )
        return norm * torch.mean(nll_loss_mat)

    def loss_sb(self, input, target, reduce=False):
        """BCE loss for batched full-adjacency reconstruction (used in ST-BrainXDL forward)."""
        temp_sum = input.sum(dim=[-1, -2])
        temp_size = input.shape[1]
        weight = (temp_size * temp_size - temp_sum) / (temp_sum + 1e-6)
        norm = (temp_size * temp_size) / ((temp_size * temp_size - temp_sum + 1e-6) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(
            input=input, target=target, reduction="none"
        )
        nll_loss = norm * weight * torch.mean(nll_loss_mat, dim=[-1, -2])
        if reduce:
            nll_loss = torch.mean(nll_loss)
        return nll_loss


class GraphClassifier(nn.Module):
    """MLP classifier operating on graph-level readout vectors."""

    def __init__(self, input_dim, layer_dims, output_dim, dropout=0.0, batch_norm=False):
        super().__init__()
        self.dropout = dropout
        self.linear = dense_vary(
            input_dim, layer_dims, output_dim,
            last_act=None, dropout=dropout, batch_norm=batch_norm,
        )

    def forward(self, input, target, sample=False):
        y_logit = self.linear(input)
        y_bce = self.loss(y_logit, target)
        with torch.no_grad():
            y_acc, y_prob, y_pred = self.acc(y_logit, target)
        y_BCE = y_bce.mean()
        y_ACC = y_acc.mean()
        if sample:
            return y_BCE, y_ACC, y_prob, y_pred
        return y_BCE, y_ACC

    def loss(self, logit, target, reduce=False):
        if logit.shape[-1] > 1:
            nll_loss = F.cross_entropy(input=logit, target=target, reduction="none")
        else:
            nll_loss = F.binary_cross_entropy_with_logits(
                input=logit.squeeze(-1), target=target.float(), reduction="none"
            )
        if reduce:
            nll_loss = torch.mean(nll_loss)
        return nll_loss

    def acc(self, logit, target, reduce=False):
        if logit.shape[-1] > 1:
            prob = F.softmax(logit, dim=-1)
            pred = torch.argmax(prob, dim=-1).float()
            acc = (target == pred).float()
        else:
            prob = torch.sigmoid(logit.squeeze(-1))
            pred = torch.round(prob)
            acc = (target == pred).float()
        if reduce:
            acc = torch.mean(acc)
            prob = torch.mean(prob)
        return acc, prob, pred


class GRU_GCN(nn.Module):
    """GRU cell where gate operations use TransformerConv instead of Linear."""

    def __init__(self, input_size, hidden_size, n_layer=1, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.weight_xz = TransformerConv(input_size, hidden_size, bias=bias)
        self.weight_hz = TransformerConv(hidden_size, hidden_size, bias=bias)
        self.weight_xr = TransformerConv(input_size, hidden_size, bias=bias)
        self.weight_hr = TransformerConv(hidden_size, hidden_size, bias=bias)
        self.weight_xh = TransformerConv(input_size, hidden_size, bias=bias)
        self.weight_hh = TransformerConv(hidden_size, hidden_size, bias=bias)

    def forward(self, input, edge_index, h):
        z_g = torch.sigmoid(
            self.weight_xz(input, edge_index) + self.weight_hz(h, edge_index)
        )
        r_g = torch.sigmoid(
            self.weight_xr(input, edge_index) + self.weight_hr(h, edge_index)
        )
        h_tilde_g = torch.tanh(
            self.weight_xh(input, edge_index) + self.weight_hh(r_g * h, edge_index)
        )
        return z_g * h + (1 - z_g) * h_tilde_g


class STBrainXDL(torch.nn.Module):
    """ST-BrainXDL spatio-temporal graph model.

    Combines a VAE prior/posterior over latent node states with a
    graph-structured GRU for dynamic brain graph classification.

    Args:
        setting (dict): Must contain 'model_params' sub-dict with keys:
            num_nodes, num_classes, x_dim, y_dim, z_dim, z_phi_dim,
            x_phi_dim, z_hidden_dim, y_hidden_dim, rnn_dim, layer_dims,
            x_hidden_dim.
    """

    def __init__(self, setting):
        super().__init__()

        model_params = setting["model_params"]
        self.recurrent = setting.get("recurrent", True)
        self.graphRNN = setting.get("graphRNN", True)

        x_dim       = model_params["x_dim"]
        z_dim       = model_params["z_dim"]
        y_dim       = model_params["y_dim"]
        x_phi_dim   = model_params["x_phi_dim"]
        z_phi_dim   = model_params["z_phi_dim"]
        z_hidden_dim = model_params["z_hidden_dim"]
        y_hidden_dim = model_params["y_hidden_dim"]
        rnn_dim     = model_params["rnn_dim"]
        layer_dims  = model_params["layer_dims"]

        self.num_nodes  = model_params["num_nodes"]
        self.num_classes = model_params["num_classes"]
        self.rnn_dim    = rnn_dim
        self.readout_mode = str(model_params.get("readout_mode", "legacy"))
        if self.readout_mode not in {"legacy", "compact_temporal"}:
            raise ValueError("readout_mode must be 'legacy' or 'compact_temporal'")
        self.stable_fc_skip = bool(model_params.get("stable_fc_skip", False))
        self.device     = get_device()
        self.EPS        = 1e-15
        self.rng_state  = torch.get_rng_state()

        # --- Input transformation ---
        self.phi_x = dense_vary(x_dim, [], x_phi_dim, last_act="relu")

        # --- Prior p(z_t | h_{t-1}) ---
        self.prior_z_hidden = GCN_vary(rnn_dim, [], z_hidden_dim, last_act="relu")
        self.prior_z_mean   = dense_vary(z_hidden_dim, [], z_dim, last_act=None)
        self.prior_z_std    = dense_vary(z_hidden_dim, [], z_dim, last_act="softplus")

        # --- Posterior q(z_t | x_t, h_{t-1}) ---
        enc_in_dim = x_phi_dim + rnn_dim if self.recurrent else x_phi_dim
        self.enc_z_hidden = GCN_vary(enc_in_dim, layer_dims, z_hidden_dim, last_act="relu")
        self.enc_z_mean   = dense_vary(z_hidden_dim, [], z_dim, last_act=None)
        self.enc_z_std    = dense_vary(z_hidden_dim, [], z_dim, last_act="softplus")

        # --- Latent transformation ---
        self.phi_z = dense_vary(z_dim, [], z_phi_dim, last_act="relu")

        # --- Recurrent state update ---
        if self.recurrent:
            rnn_in_dim = x_phi_dim + z_phi_dim
            if self.graphRNN:
                self.rnn_cell = GRU_GCN(rnn_in_dim, rnn_dim)
            else:
                self.rnn_cell = recurrent_cell(rnn_in_dim, rnn_dim, "gru")

        # --- Decoders ---
        self.dec_graph = GraphDecoder()

        # x decoder is currently disabled in the forward pass.
        # dec_in_dim = z_phi_dim + rnn_dim if self.recurrent else z_phi_dim
        # self.dec_x_hidden = GCN_vary(dec_in_dim, [], x_hidden_dim, last_act='relu')
        # self.dec_x_mean   = dense_vary(x_hidden_dim, [], x_dim, last_act=None)
        # self.dec_x_std    = dense_vary(x_hidden_dim, [], x_dim, last_act='softplus')

        # --- Classifier ---
        node_readout_dim = z_phi_dim + rnn_dim if self.recurrent else z_phi_dim
        legacy_readout_dim = (
            self.num_nodes * (z_phi_dim + rnn_dim)
            if self.recurrent
            else self.num_nodes * z_phi_dim
        )
        if self.readout_mode == "compact_temporal":
            projection_dim = int(model_params.get("readout_projection_dim", 32))
            self.readout_projection = nn.Sequential(
                nn.Linear(2 * node_readout_dim, projection_dim),
                nn.ReLU(),
                nn.LayerNorm(projection_dim),
            )
            readout_dim = 3 * projection_dim
        else:
            self.readout_projection = None
            readout_dim = legacy_readout_dim

        if self.stable_fc_skip:
            stable_dim = int(model_params.get("stable_fc_dim", 32))
            edge_dim = self.num_nodes * (self.num_nodes - 1) // 2
            self.stable_fc_projector = nn.Sequential(
                nn.Linear(edge_dim, stable_dim),
                nn.ReLU(),
                nn.LayerNorm(stable_dim),
            )
            self.register_buffer(
                "stable_fc_upper_indices",
                torch.triu_indices(self.num_nodes, self.num_nodes, offset=1),
                persistent=False,
            )
            readout_dim += stable_dim
        else:
            self.stable_fc_projector = None
        self.classifier = GraphClassifier(
            readout_dim,
            y_hidden_dim,
            y_dim,
            dropout=float(model_params.get("classifier_dropout", 0.5)),
            batch_norm=True,
        )

    def apply_device_split(self, dense_device: torch.device, sparse_on_cpu: bool = True):
        """Place PyG/graph modules on CPU, dense modules on dense_device.

        Must be called after any model.to(...) to re-enforce the split.
        This is required for MPS (Apple Silicon) which does not support
        all sparse/PyG scatter operations.
        """
        self.device = dense_device
        self.sparse_on_cpu = sparse_on_cpu
        graph_device = torch.device("cpu") if sparse_on_cpu else dense_device

        # Graph modules → CPU
        self.enc_z_hidden.to(graph_device)
        self.prior_z_hidden.to(graph_device)
        if getattr(self, "recurrent", False) and getattr(self, "graphRNN", False):
            self.rnn_cell.to(graph_device)

        # Dense modules → dense_device
        self.phi_x.to(dense_device)
        self.enc_z_mean.to(dense_device)
        self.enc_z_std.to(dense_device)
        self.prior_z_mean.to(dense_device)
        self.prior_z_std.to(dense_device)
        self.phi_z.to(dense_device)
        self.dec_graph.to(dense_device)
        if self.readout_projection is not None:
            self.readout_projection.to(dense_device)
        if self.stable_fc_projector is not None:
            self.stable_fc_projector.to(dense_device)
        self.classifier.to(dense_device)

    def forward(self, graphs, setting, sample=False):
        """Process a sequence of graphs and return ST-BrainXDL losses + classifier readout.

        Args:
            graphs:  List of torch_geometric.data.Batch objects, one per time step.
            setting: Training config dict (must include 'variational', 'device',
                     'sparse_on_cpu').
            sample:  If True, also return sampled latents and adjacency matrices.

        Returns:
            (x_NLL, z_KLD, a_NLL, Readout, Target) or with sample=True appends
            (z_Sample, adj_Sample, h_Sample, zh_Sample).
        """
        variational = setting["variational"]
        self.device = torch.device(setting.get("device", getattr(self, "device", "cpu")))
        graph_device = (
            torch.device("cpu") if setting.get("sparse_on_cpu", True) else self.device
        )

        Mask = []; Last = []
        x_NLL = []; z_KLD = []; adj_NLL = []
        Readout = []; Target = []; StableFC = []
        if sample:
            z_Sample = []; adj_Sample = []; h_Sample = []; zh_Sample = []

        batch_size = graphs[0].num_graphs
        h = torch.zeros(
            batch_size * self.num_nodes, self.rnn_dim
        ).to(self.device, dtype=torch.float)

        for step, graph in enumerate(graphs):
            x          = graph.x.to(self.device)
            y          = graph.y.to(self.device)
            mask       = (graph.pad == False).to(self.device)
            last       = graph.last.to(self.device)
            edge_index = graph.edge_index.to(graph_device)
            batch      = graph.batch.to(graph_device)
            adj        = graph.adj.to(graph_device)

            x_phi = self.phi_x(x)

            # Posterior q(z_t | x_t, h_{t-1})
            enc_in = torch.cat([x_phi, h], dim=-1) if self.recurrent else x_phi
            self.enc_z_hidden.to(graph_device)
            enc_in_cpu = enc_in.to(graph_device)
            z_enc_hidden_cpu = self.enc_z_hidden(enc_in_cpu, edge_index)
            z_enc_hidden = z_enc_hidden_cpu.to(self.device)
            z_enc_mean_sb = self.sep_batch(self.enc_z_mean(z_enc_hidden), batch.to(self.device))

            if variational:
                z_enc_std_sb = self.sep_batch(
                    self.enc_z_std(z_enc_hidden), batch.to(self.device)
                )
                # Prior p(z_t | h_{t-1})
                if self.recurrent:
                    self.prior_z_hidden.to(graph_device)
                    z_prior_hidden_cpu = self.prior_z_hidden(h.to(graph_device), edge_index)
                    z_prior_hidden = z_prior_hidden_cpu.to(self.device)
                    z_prior_mean_sb = self.sep_batch(
                        self.prior_z_mean(z_prior_hidden), batch.to(self.device)
                    )
                    z_prior_std_sb = self.sep_batch(
                        self.prior_z_std(z_prior_hidden), batch.to(self.device)
                    )
                else:
                    z_prior_mean_sb = torch.zeros_like(z_enc_mean_sb).to(self.device)
                    z_prior_std_sb  = torch.zeros_like(z_enc_mean_sb).to(self.device)

                z_kld_sb = self.kld_normal_sb(
                    z_enc_mean_sb, z_enc_std_sb,
                    z_prior_mean_sb, z_prior_std_sb,
                    reduce=False,
                )
                if self.training:
                    z_sample_sb = self.reparameterize_normal(z_enc_mean_sb, z_enc_std_sb)
                else:
                    z_sample_sb = z_enc_mean_sb
            else:
                z_kld_sb    = torch.zeros(batch_size).to(self.device)
                z_sample_sb = z_enc_mean_sb

            z_phi = self.phi_z(torch.flatten(z_sample_sb, end_dim=1))

            h_sb = self.sep_batch(h, batch)
            adj_in_sb  = torch.cat([z_sample_sb, h_sb], dim=-1) if self.recurrent else z_sample_sb
            adj_dec_sb = self.dec_graph.forward_sb(adj_in_sb)
            adj_sb     = self.sep_batch(adj, batch).to(self.device)
            adj_nll_sb = self.dec_graph.loss_sb(adj_dec_sb, adj_sb, reduce=False)

            if self.recurrent:
                rnn_in = torch.cat([x_phi, z_phi], dim=-1)
                if self.graphRNN:
                    self.rnn_cell.to(graph_device)
                    h_new_cpu = self.rnn_cell(
                        rnn_in.to(graph_device), edge_index, h.to(graph_device)
                    )
                    h = h_new_cpu.to(self.device)
                else:
                    h = self.rnn_cell(rnn_in, h)

            x_NLL_step = torch.zeros(1).to(self.device)  # x decoder disabled

            z_readout_sb    = self.phi_z(z_enc_mean_sb)
            readout_sb      = torch.cat([z_readout_sb, h_sb], dim=-1) if self.recurrent else z_readout_sb
            if self.readout_mode == "legacy":
                readout_step = readout_sb.flatten(start_dim=1, end_dim=2)
            else:
                node_pooled = torch.cat(
                    [readout_sb.mean(dim=1), readout_sb.max(dim=1).values], dim=-1
                )
                readout_step = self.readout_projection(node_pooled)
            if self.stable_fc_skip:
                x_sb = self.sep_batch(x, batch.to(self.device))
                upper = self.stable_fc_upper_indices
                StableFC.append(x_sb[:, upper[0], upper[1]])

            Mask.append(mask)
            Last.append(last)
            x_NLL.append(x_NLL_step)
            z_KLD.append(z_kld_sb)
            adj_NLL.append(adj_nll_sb)
            Readout.append(readout_step)
            Target.append(y)

            if sample:
                z_Sample.append(z_sample_sb)
                adj_Sample.append(torch.sigmoid(adj_dec_sb))
                h_Sample.append(h_sb)
                zh_Sample.append(adj_in_sb)

        Mask   = torch.stack(Mask)
        SeqLen = Mask.sum(dim=0)

        x_NLL_final   = ((torch.stack(x_NLL)   * Mask).sum(dim=0) / SeqLen).mean()
        z_KLD_final   = ((torch.stack(z_KLD)   * Mask).sum(dim=0) / SeqLen).mean()
        adj_NLL_final = ((torch.stack(adj_NLL) * Mask).sum(dim=0) / SeqLen).mean()

        Last = torch.stack(Last)
        assert Last.sum() == batch_size
        readout_stack = torch.stack(Readout)
        if self.readout_mode == "legacy":
            Readout_final = readout_stack[Last]
            Target_final = torch.stack(Target)[Last]
        else:
            mask_float = Mask.to(readout_stack.dtype).unsqueeze(-1)
            temporal_mean = (readout_stack * mask_float).sum(dim=0) / SeqLen.unsqueeze(-1)
            temporal_max = readout_stack.masked_fill(~Mask.unsqueeze(-1), -torch.inf).max(dim=0).values
            batch_indices = torch.arange(batch_size, device=self.device)
            last_indices = SeqLen.long() - 1
            temporal_final = readout_stack[last_indices, batch_indices]
            Readout_final = torch.cat(
                [temporal_mean, temporal_max, temporal_final], dim=-1
            )
            Target_final = torch.stack(Target)[last_indices, batch_indices]

        if self.stable_fc_skip:
            stable_stack = torch.stack(StableFC)
            stable_mean = (stable_stack * Mask.to(stable_stack.dtype).unsqueeze(-1)).sum(dim=0)
            stable_mean = stable_mean / SeqLen.unsqueeze(-1)
            Readout_final = torch.cat(
                [Readout_final, self.stable_fc_projector(stable_mean)], dim=-1
            )

        if sample:
            z_Sample   = stack_truncate(torch.stack(z_Sample),   Mask)
            adj_Sample = stack_truncate(torch.stack(adj_Sample), Mask)
            h_Sample   = stack_truncate(torch.stack(h_Sample),   Mask)
            zh_Sample  = stack_truncate(torch.stack(zh_Sample),  Mask)
            return (
                x_NLL_final, z_KLD_final, adj_NLL_final,
                Readout_final, Target_final,
                z_Sample, adj_Sample, h_Sample, zh_Sample,
            )
        return x_NLL_final, z_KLD_final, adj_NLL_final, Readout_final, Target_final

    def sep_batch(self, input, batch):
        """Unpack flat (TotalNodes x F) tensor into (Batch x Nodes x F)."""
        if batch.device != input.device:
            batch = batch.to(input.device)
        output, _ = to_dense_batch(input, batch)
        return output

    def reparameterize_normal(self, mean, std):
        eps = torch.randn(mean.shape).to(self.device, dtype=torch.float)
        return mean + torch.mul(eps, std)

    def kld_normal_sb(self, mean_q, std_q, mean_p, std_p, reduce=False):
        """KL divergence between two diagonal Gaussians, batched over nodes."""
        kld_element = (
            2 * (torch.log(std_p + self.EPS) - torch.log(std_q + self.EPS))
            + (torch.pow(std_q + self.EPS, 2) + torch.pow(mean_q - mean_p, 2))
            / torch.pow(std_p + self.EPS, 2)
            - 1
        )
        kld = (0.5 / self.num_nodes) * torch.sum(kld_element, dim=[-1, -2])
        if reduce:
            kld = torch.mean(kld)
        return kld

    # --- Unused but preserved from legacy ---

    def kld_mvnormal(self, mean_q, node_sqm_q, feat_std_q, mean_p, std_p):
        """Multivariate normal KLD (not called in current forward pass)."""
        n_size, m_size = mean_q.shape[1:]
        node_cov_q = node_sqm_q @ torch.transpose(node_sqm_q, dim0=-2, dim1=-1)
        nll_q = (
            n_size * (2 * torch.log(feat_std_q + self.EPS)).sum(dim=-1)
            + m_size * torch.log(det(node_cov_q) + self.EPS)
            + n_size * m_size
        )
        nll_p_det = (2 * torch.log(std_p + self.EPS)).sum(dim=[-1, -2])
        UpVp = (std_p.pow(2) + self.EPS).pow(-1)
        UqVq = (
            torch.diagonal(node_cov_q, dim1=-2, dim2=-1).unsqueeze(-1)
            @ feat_std_q.pow(2).unsqueeze(-2)
        )
        nll_p_tr  = (UqVq * UpVp).sum(dim=[-1, -2])
        Xd        = (mean_q - mean_p) * std_p
        nll_p_wls = (Xd * Xd).sum(dim=[-1, -2])
        nll_p     = nll_p_det + nll_p_tr + nll_p_wls
        kld = (0.5 / (n_size * m_size)) * torch.mean(nll_p - nll_q, dim=0)
        return kld if kld > 0 else torch.zeros(1).to(self.device, dtype=torch.float)

    def nll_normal_sb(self, x, mean, std, reduce=False):
        """Normal NLL for x reconstruction (not used in current forward pass)."""
        constant = math.log(2 * math.pi)
        xd = x - mean
        nll_element = (
            2 * torch.log(std + self.EPS)
            + torch.div(xd, std + self.EPS).pow(2)
            + constant
        )
        nll = (0.5 / (self.num_nodes * mean.shape[-1])) * torch.sum(
            nll_element, dim=[-1, -2]
        )
        if reduce:
            nll = torch.mean(nll)
        return nll

    def edge_recon_loss(self, z, batch, pos_edge_index, neg_edge_index=None):
        """Edge reconstruction loss for link prediction (not used in forward pass)."""
        num_pos = pos_edge_index.shape[-1]
        pos_dec = self.dec_graph(z, pos_edge_index)
        norm    = (num_pos * num_pos) / float((num_pos * num_pos - pos_dec.sum()) * 2)
        pos_loss = -torch.log(pos_dec + self.EPS).mean()
        if neg_edge_index is None:
            neg_edge_index = batched_negative_sampling(pos_edge_index, batch)
        neg_dec  = self.dec_graph(z, neg_edge_index)
        neg_loss = -torch.log(1 - neg_dec + self.EPS).mean()
        return norm * (pos_loss + neg_loss)


VGRNN = STBrainXDL
