"""Connect CNN features to a Transformer and classify using a single forward pass."""

from torch import nn

from .cnn_encoder import CNNEncoder
from .transformer_encoder import TransformerEncoder


class ClassificationHead(nn.Module):
    def __init__(self, d_model=64, num_classes=5, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(512, eps=1e-6)
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(256, eps=1e-6)
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.residual_projection = nn.Linear(d_model, 128)
        self.norm3 = nn.LayerNorm(128, eps=1e-6)
        self.output = nn.Linear(128, num_classes)

    def forward(self, tokens):
        pooled = tokens.mean(dim=1)
        x = self.norm1(self.dropout1(self.relu1(self.fc1(pooled))))
        x = self.norm2(self.dropout2(self.relu2(self.fc2(x))))
        x = self.relu3(self.fc3(x))
        return self.output(self.norm3(x + self.residual_projection(pooled)))


class ECGClassifier(nn.Module):
    num_classes = 5

    def __init__(self, more_cnn=False, transformer_blocks=2, dropout=0.2, max_length=100):
        super().__init__()
        self.cnn_encoder = CNNEncoder(dropout=dropout, more_cnn=more_cnn)
        self.transformer_encoder = TransformerEncoder(
            d_model=64, num_heads=4, ff_dim=256, num_layers=transformer_blocks,
            dropout=dropout, max_length=max_length,
        )
        self.head = ClassificationHead(d_model=64, num_classes=self.num_classes, dropout=dropout)

    def forward(self, signals):
        """Return raw logits [B, 5] from ECG [B, 1, 300 or 400]."""
        features = self.cnn_encoder(signals)
        tokens = self.transformer_encoder(features.transpose(1, 2))
        return self.head(tokens)
