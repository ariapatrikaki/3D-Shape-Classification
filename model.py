import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        """
        
        DEFINE YOUR NETWORK HERE
        
        """

        # Conv layer with 8 filters, 7x7 kernel
        # Disable bias since batch normalization will follow
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=7, stride=1, padding=0, bias=False)
        
        # Batch normalization with num_features=8 (number of output channels from conv1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        
        # Leaky ReLU with negative slope of 0.01
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        
        # Max pooling with 2x2 window and stride 2
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Depthwise convolution layer with 8 filters, 7x7 kernel, groups=8 for depthwise
        # Disable bias since batch normalization will follow
        self.depthwise_conv = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=7, stride=2, padding=0, groups=8, bias=False)
        
        # Batch normalization with num_features=8 (number of output channels from depthwise_conv)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        
        # Leaky ReLU with negative slope of 0.01
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.01)
        
        # Max pooling with 2x2 window and stride 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Pointwise convolution layer with 16 filters, 1x1 kernel
        # Enable bias since there's no batch norm following this layer
        self.pointwise_conv = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Depthwise convolution layer with 16 filters, 7x7 kernel, groups=16 for depthwise
        # Disable bias since batch normalization will follow
        self.depthwise_conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=7, stride=1, padding=0, groups=16, bias=False)
        
        # Batch normalization with num_features=16 (number of output channels from depthwise_conv2)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        
        # Leaky ReLU with negative slope of 0.01
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.01)
        
        # Max pooling with 2x2 window and stride 2
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Pointwise convolution layer with 32 filters, 1x1 kernel
        # Disable bias since there's no batch norm and we'll add it later
        self.pointwise_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer implemented as 1x1 convolution with num_classes output
        # Enable bias for the fully connected layer
        self.fc = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
        # Initialize weights and biases
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights according to the following scheme:
        - Kaiming (He) uniform for conv layers followed by leaky ReLU
        - Xavier uniform for pointwise convolution and FC layers
        - All biases initialized to 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Check if this layer is followed by leaky ReLU (conv layers followed by leaky ReLU)
                # For regular conv and depthwise conv with leaky ReLU: use Kaiming uniform
                if hasattr(self, 'conv1') and m == self.conv1:
                    # conv1 is followed by BN and leaky ReLU
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                elif hasattr(self, 'depthwise_conv') and m == self.depthwise_conv:
                    # depthwise_conv is followed by BN and leaky ReLU
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                elif hasattr(self, 'depthwise_conv2') and m == self.depthwise_conv2:
                    # depthwise_conv2 is followed by BN and leaky ReLU
                    nn.init.kaiming_uniform_(m.weight, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                elif hasattr(self, 'pointwise_conv') and m == self.pointwise_conv:
                    # pointwise_conv is not followed by leaky ReLU: use Xavier uniform
                    nn.init.xavier_uniform_(m.weight)
                elif hasattr(self, 'pointwise_conv2') and m == self.pointwise_conv2:
                    # pointwise_conv2 is not followed by leaky ReLU: use Xavier uniform
                    nn.init.xavier_uniform_(m.weight)
                elif hasattr(self, 'fc') and m == self.fc:
                    # fc (fully connected) is not followed by leaky ReLU: use Xavier uniform
                    nn.init.xavier_uniform_(m.weight)
                
                # Initialize bias to 0 if it exists
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """

        DEFINE YOUR FORWARD PASS HERE

        """

        # Apply first block: conv1, bn1, leaky_relu, maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.maxpool(out)
        
        # Apply depthwise separable convolution block: depthwise_conv, bn2, leaky_relu2, maxpool2, pointwise_conv
        out = self.depthwise_conv(out)
        out = self.bn2(out)
        out = self.leaky_relu2(out)
        out = self.maxpool2(out)
        out = self.pointwise_conv(out)
        
        # Apply second depthwise separable convolution block: depthwise_conv2, bn3, leaky_relu3, maxpool3, pointwise_conv2
        out = self.depthwise_conv2(out)
        out = self.bn3(out)
        out = self.leaky_relu3(out)
        out = self.maxpool3(out)
        out = self.pointwise_conv2(out)
        
        # Apply adaptive average pooling to get 1x1 spatial dimensions
        out = self.adaptive_avgpool(out)
        
        # Apply fully connected layer (implemented as 1x1 convolution)
        out = self.fc(out)
        
        # Flatten to get class scores
        out = out.view(out.size(0), -1)
        
        return out
        

class TinyViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int = 112,
        patch_size: int = 8,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        grid = img_size // patch_size
        self.num_patches = grid * grid
        
        # 1) Patch embedding layer: conv layer with 64 filters, 8x8 kernel, stride 8
        # This projects each 8x8 patch to a 64-dimensional embedding
        self.patch_embed = nn.Conv2d(in_channels=1, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        
        # 2) Learnable positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3) Learnable class token embedding
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 4) Transformer encoder with 2 layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim,
            activation='gelu',
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 5) Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # 6) Classification head (linear layer)
        self.classification_head = nn.Linear(embed_dim, num_classes, bias=True)
        
        # Initialize weights and embeddings
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights:
        - Patch embedding and linear layer: Xavier uniform
        - Biases: 0
        - Positional embeddings and class token: truncated Gaussian with std 0.02
        """
        # Initialize patch embedding layer
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.constant_(self.patch_embed.bias, 0)
        
        # Initialize classification head
        nn.init.xavier_uniform_(self.classification_head.weight)
        if self.classification_head.bias is not None:
            nn.init.constant_(self.classification_head.bias, 0)
        
        # Initialize positional embeddings and class token with truncated Gaussian (std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)
    
    def forward(self, x):
        """
        Forward pass for Vision Transformer
        """
        B = x.size(0)  # batch size
        
        # 1) Patch embedding: convert image to patch embeddings
        # Input: [B, 1, 112, 112]
        # Output after conv: [B, 64, 8, 8]
        x = self.patch_embed(x)  # [B, embed_dim, grid, grid]
        
        # Flatten spatial dimensions to get tokens
        # [B, embed_dim, grid, grid] -> [B, embed_dim, num_patches]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        
        # Transpose to [B, num_patches, embed_dim]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # 2) Add positional embeddings
        x = x + self.pos_embed  # [B, num_patches, embed_dim]
        
        # 3) Prepend class token
        class_tokens = self.class_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([class_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # 4) Pass through transformer encoder
        x = self.transformer_encoder(x)  # [B, num_patches+1, embed_dim]
        
        # 5) Extract class token output and apply layer normalization
        x = x[:, 0, :]  # [B, embed_dim] - take only class token
        x = self.layer_norm(x)  # [B, embed_dim]
        
        # 6) Classification head
        out = self.classification_head(x)  # [B, num_classes]
        
        return out        