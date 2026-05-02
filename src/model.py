import torch
import torch.nn as nn
import torch.nn.functional as F  


class ConvBlock(nn.Module):
    """Two consecutive Conv3D → BatchNorm → ReLU blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """ConvBlock followed by MaxPool — returns both for skip connection."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)   # save for skip connection
        down = self.pool(skip)
        return skip, down


class DecoderBlock(nn.Module):
    """Upsample → concatenate skip → ConvBlock."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        
        # --- THE U-NET PADDING FIX ---
        # Calculate the dimension differences
        diffZ = skip.size()[2] - x.size()[2]
        diffY = skip.size()[3] - x.size()[3]
        diffX = skip.size()[4] - x.size()[4]
        
        # Pad x to perfectly match the skip connection's shape
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2,
                      diffZ // 2, diffZ - diffZ // 2])
        # -----------------------------
        
        x = torch.cat([x, skip], dim=1)  # concatenate along channel dim
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for brain tumor segmentation.
    Input:  [B, 4, 128, 128, 128]  (4 MRI modalities)
    Output: [B, 4, 128, 128, 128]  (4 classes: background + 3 tumor regions)
    """
    def __init__(self, in_channels=4, out_channels=4, features=[32, 64, 128, 256]):
        super().__init__()

        # Encoder
        self.encoders = nn.ModuleList()
        prev_channels = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(prev_channels, feat))
            prev_channels = feat

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        # Decoder (reverse order)
        self.decoders = nn.ModuleList()
        decoder_features = list(reversed(features))
        prev_channels = features[-1] * 2
        for feat in decoder_features:
            self.decoders.append(DecoderBlock(prev_channels, feat))
            prev_channels = feat

        # Final 1x1 conv → class scores
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path — collect skip connections
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path — use skip connections in reverse
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        return self.final_conv(x)


def get_model(config):
    model = UNet3D(
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        features=config["model"]["features"],
    )
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    return trainable