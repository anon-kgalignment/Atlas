# sage/modules/models/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFusion(nn.Module):
    """
    Adaptive three-signal fusion module.

    Combines DICE, LaBSE, and GAT representations
    with learned per-entity weights.

    For each entity the gate network looks at all
    three signals and decides how much to trust each:

        w1 = trust in DICE  (relational geometry)
        w2 = trust in LaBSE (semantic meaning)
        w3 = trust in GAT   (structural + cross-KG)

        Z = w1 * proj(DICE) + w2 * proj(LaBSE) + w3 * GAT

    The weights are different for each entity and
    learned automatically during training.

    Args:
        dice_dim  : DICE embedding dimension (256)
        labse_dim : LaBSE embedding dimension (768)
        hidden_dim: output dimension (256)
        dropout   : dropout rate
    """

    def __init__(
        self,
        dice_dim  : int,
        labse_dim : int,
        hidden_dim: int,
        dropout   : float = 0.1,
    ):
        super().__init__()

        self.dice_dim      = dice_dim
        self.labse_dim     = labse_dim
        self.hidden_dim    = hidden_dim
        self.entropy_weight = 0.01

        # ── Project each signal to hidden_dim ────

        # DICE: 256 → 256
        self.proj_dice = nn.Sequential(
            nn.Linear(dice_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # LaBSE: 768 → 256
        self.proj_labse = nn.Sequential(
            nn.Linear(labse_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        self.proj_gat = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        gate_input_dim = hidden_dim * 3

        self.gate_fc1    = nn.Linear(gate_input_dim, hidden_dim)
        self.gate_relu   = nn.ReLU()
        self.gate_drop   = nn.Dropout(dropout)
        self.gate_fc2    = nn.Linear(hidden_dim, 3)

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._init_gate_equal()

    def _init_gate_equal(self):
        """
        Initialize gate_fc2 to zero weights and biases
        so softmax produces equal weights [0.33, 0.33, 0.33]
        at the start of training.
        """
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)

    def _compute_gate(self, gate_input: torch.Tensor) -> torch.Tensor:
        """
        Run the gate network manually through each layer.

        Args:
            gate_input: [n, hidden_dim * 3]

        Returns:
            gate weights [n, 3] summing to 1
        """
        x = self.gate_fc1(gate_input)
        x = self.gate_relu(x)
        x = self.gate_drop(x)
        x = self.gate_fc2(x)
        return F.softmax(x, dim=1)

    def forward(
        self,
        E : torch.Tensor,
        P : torch.Tensor,
        H : torch.Tensor,
    ):
        """
        Forward pass.

        Args:
            E : DICE embeddings  [n_entities, dice_dim]
            P : LaBSE embeddings [n_entities, labse_dim]
            H : GAT embeddings   [n_entities, hidden_dim]

        Returns:
            Z           : fused embeddings [n_entities, hidden_dim]
            gate_weights: [n_entities, 3]
                          [:, 0] = DICE weights
                          [:, 1] = LaBSE weights
                          [:, 2] = GAT weights
        """
        e_proj = self.proj_dice(E)    # [n, hidden_dim]
        p_proj = self.proj_labse(P)   # [n, hidden_dim]
        h_proj = self.proj_gat(H)     # [n, hidden_dim]

        gate_input   = torch.cat([e_proj, p_proj, h_proj], dim=1)
        gate_weights = self._compute_gate(gate_input)  # [n, 3]

        w_dice  = gate_weights[:, 0:1]  # [n, 1]
        w_labse = gate_weights[:, 1:2]  # [n, 1]
        w_gat   = gate_weights[:, 2:3]  # [n, 1]

        Z = (
            w_dice  * e_proj +
            w_labse * p_proj +
            w_gat   * h_proj
        )

        Z = self.output_proj(Z)

        return Z, gate_weights

    def diversity_loss(
        self,
        gate_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entropy regularization to prevent gate collapse.

        Encourages the gate to use all three signals
        rather than always picking one.

        Args:
            gate_weights: [n_entities, 3] from forward()

        Returns:
            scalar loss value (small positive number)
        """

        entropy = -(
            gate_weights * torch.log(gate_weights + 1e-8)
        ).sum(dim=1)

        loss = -entropy.mean()

        return loss * self.entropy_weight