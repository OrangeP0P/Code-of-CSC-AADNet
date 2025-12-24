import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CSC_AADNet(nn.Module):
    """
    Expected input: x with shape [batch, chans, samples] (e.g., [B, 32, T])
    - chans should be 32 for your setup, but the module is configurable.
    - samples can be any length; the network uses adaptive pooling to handle variable T.

    Args:
        n_classes (int): number of output classes.
        chans (int): number of EEG channels (default: 32).
        F1 (int): number of temporal filters in the first conv layer.
        D (int): depthwise multiplier.
        kernel_len (int): temporal kernel length (in samples at your sampling rate).
        dropout (float): dropout probability.
        pool_size (int): temporal pooling size after conv blocks.
        separable_kernel_len (int | None): kernel length for separable conv. If None, uses kernel_len // 2.
        use_batchnorm (bool): whether to use BatchNorm.
    """

    def __init__(
        self,
        n_classes: int =2,
        chans: int = 32,
        F1: int = 8,
        D: int = 2,
        kernel_len: int = 64,
        dropout: float = 0.25,
        pool_size: int = 4,
        separable_kernel_len: int | None = None,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.chans = chans
        self.F1 = F1
        self.D = D
        self.F2 = F1 * D
        self.dropout_p = dropout
        self.pool_size = pool_size
        self.use_batchnorm = use_batchnorm
        sep_k = separable_kernel_len or max(8, kernel_len // 2)

        # Block 1: Temporal Convolution
        # Input will be reshaped to [B, 1, C, T] then conv across time axis only
        self.conv_time = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_len),
            padding=(0, kernel_len // 2),  # 'same' padding for time dim
            bias=False,
        )
        self.bn_time = nn.BatchNorm2d(F1) if use_batchnorm else nn.Identity()

        # Block 2: Depthwise Convolution (spatial filtering across channels)
        self.conv_depth = nn.Conv2d(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(chans, 1),
            groups=F1,  # depthwise over F1 maps
            bias=False,
        )
        self.bn_depth = nn.BatchNorm2d(F1 * D) if use_batchnorm else nn.Identity()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool_size))
        self.drop1 = nn.Dropout(p=dropout)

        # Block 3: Separable Convolution (depthwise temporal + pointwise)
        # Depthwise temporal
        self.conv_separable_depth = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, sep_k),
            groups=F1 * D,
            padding=(0, sep_k // 2),
            bias=False,
        )
        # Pointwise
        self.conv_separable_point = nn.Conv2d(
            in_channels=F1 * D,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn_sep = nn.BatchNorm2d(self.F2) if use_batchnorm else nn.Identity()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool_size))
        self.drop2 = nn.Dropout(p=dropout)

        # Global average pooling to remove dependence on T
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Linear(self.F2, n_classes)

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape [B, chans, samples]
        Returns:
            logits: Tensor of shape [B, n_classes]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input of shape [B, C, T], got {tuple(x.shape)}")
        if x.size(1) != self.chans:
            raise ValueError(f"Expected {self.chans} channels, got {x.size(1)}")

        x = x.unsqueeze(1)

        # Block 1
        x = self.conv_time(x)
        x = self.bn_time(x)
        x = F.elu(x)

        # Block 2
        x = self.conv_depth(x)
        x = self.bn_depth(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 3
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        x = self.bn_sep(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Global Average Pooling
        x = self.gap(x)  # [B, F2, 1, 1]
        x = x.flatten(start_dim=1)  # [B, F2]

        logits = self.classifier(x)
        return logits
