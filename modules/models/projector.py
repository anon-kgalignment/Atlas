
import torch
import torch.nn as nn
import torch.nn.functional as F


class KGProjector(nn.Module):
    """
    KG-specific projector that maps fused entity
    embeddings into the shared alignment space.

    Each KG gets its own projector because:
    - KG1 and KG2 were trained independently by DICE
    - They have different internal geometric structures
    - One shared projector cannot optimally handle both
    - Each KG needs its own learned correction

    Architecture:
        Z [hidden_dim]
        → Linear(hidden_dim, hidden_dim) + GELU
        → Linear(hidden_dim, hidden_dim)
        → residual connection: output = input + transform(input)
        → LayerNorm
        → L2 normalize

    Args:
        hidden_dim : input and output dimension (256)
        dropout    : dropout rate
    """

    def __init__(
        self,
        hidden_dim : int,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.drop = nn.Dropout(dropout)
        
        self.residual_weight = nn.Parameter(
            torch.zeros(1)
        )

        self._init_weights()

    def _init_weights(self):
    
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
        nn.init.zeros_(self.fc1.bias)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            Z : fused embeddings [n_entities, hidden_dim]

        Returns:
            aligned embeddings [n_entities, hidden_dim]
            L2 normalized
        """
    
        transformed = self.fc1(Z)
        transformed = F.gelu(transformed)
        transformed = self.drop(transformed)
        transformed = self.fc2(transformed)

        weight = torch.sigmoid(self.residual_weight)
        out = Z + weight * transformed

        out = self.norm(out)
        out = F.normalize(out, dim=1)

        return out


class DualProjector(nn.Module):
    """
    Dual projector with one KGProjector per KG.
    Args:
        hidden_dim : embedding dimension (256)
        dropout    : dropout rate
    """

    def __init__(
        self,
        hidden_dim : int,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.proj_kg1 = KGProjector(hidden_dim, dropout)
        self.proj_kg2 = KGProjector(hidden_dim, dropout)

    def forward_kg1(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Project KG1 fused embeddings into alignment space.

        Args:
            Z : KG1 fused embeddings [n, hidden_dim]

        Returns:
            aligned KG1 embeddings [n, hidden_dim]
            L2 normalized
        """
        return self.proj_kg1(Z)

    def forward_kg2(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Project KG2 fused embeddings into alignment space.

        Args:
            Z : KG2 fused embeddings [n, hidden_dim]

        Returns:
            aligned KG2 embeddings [n, hidden_dim]
            L2 normalized
        """
        return self.proj_kg2(Z)

    def forward(
        self,
        Z          : torch.Tensor,
        kg1_ids    : torch.Tensor,
        kg2_ids    : torch.Tensor,
        n_entities : int,
    ) -> torch.Tensor:
        """
        Project all entities using their KG-specific projector.

        Entities from KG1 go through proj_kg1.
        Entities from KG2 go through proj_kg2.

        Args:
            Z          : all fused embeddings [n_entities, hidden_dim]
            kg1_ids    : indices of KG1 entities in Z
            kg2_ids    : indices of KG2 entities in Z
            n_entities : total number of entities

        Returns:
            A : aligned embeddings [n_entities, hidden_dim]
                same ordering as Z
                L2 normalized
        """
        A = torch.zeros_like(Z)

        if len(kg1_ids) > 0:
            kg1_ids_t = torch.tensor(
                list(kg1_ids),
                dtype=torch.long,
                device=Z.device
            )
            Z_kg1    = Z[kg1_ids_t]
            A_kg1    = self.proj_kg1(Z_kg1)
            A[kg1_ids_t] = A_kg1

        if len(kg2_ids) > 0:
            kg2_ids_t = torch.tensor(
                list(kg2_ids),
                dtype=torch.long,
                device=Z.device
            )
            Z_kg2    = Z[kg2_ids_t]
            A_kg2    = self.proj_kg2(Z_kg2)
            A[kg2_ids_t] = A_kg2

        return A