# Custom Lane U-Net Architecture

Input `3×48×48` RGB → output `1×48×48` logits (sigmoid > 0.5 → binary lane mask). 482,737 parameters, trained from scratch.

```mermaid
flowchart TD
    IN["Input 3×48×48"] --> E1["Enc1: ConvBlock 3→16 (16×48×48)"]
    E1 --> P1["MaxPool 2×2"] --> E2["Enc2: ConvBlock 16→32 (32×24×24)"]
    E2 --> P2["MaxPool 2×2"] --> E3["Enc3: ConvBlock 32→64 (64×12×12)"]
    E3 --> P3["MaxPool 2×2"] --> B["Bottleneck: ConvBlock 64→128 (128×6×6)"]
    B --> U3["ConvTranspose 2×2, 128→64 (64×12×12)"]
    U3 --> D3["Concat Enc3 → ConvBlock 128→64"]
    D3 --> U2["ConvTranspose 2×2, 64→32 (32×24×24)"]
    U2 --> D2["Concat Enc2 → ConvBlock 64→32"]
    D2 --> U1["ConvTranspose 2×2, 32→16 (16×48×48)"]
    U1 --> D1["Concat Enc1 → ConvBlock 32→16"]
    D1 --> H["Conv 1×1, 16→1 (1×48×48)"]
    H --> OUT["Sigmoid > 0.5 → Binary mask 48×48"]
    E1 -. skip .-> D1
    E2 -. skip .-> D2
    E3 -. skip .-> D3
```

`ConvBlock` = `[Conv3×3 (no bias) → BatchNorm → ReLU] × 2`.
