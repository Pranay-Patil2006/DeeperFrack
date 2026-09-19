# PolyFuse V2 Final Report

## Verification Results

| Metric | Naive (SLC) | PolyFuse V2 (Fused) | Reduction |
|:---|---:|---:|---:|
| Layer 0 Energy | 766320000.00 pJ | - | - |
| Layer 1 Energy | 19868670000.00 pJ | - | - |
| Layer 2 Energy | 6534050000.00 pJ | - | - |
| Layer 3 Energy | 30686040000.00 pJ | - | - |
| Layer 4 Energy | 1484770000.00 pJ | - | - |
| Stack 0 Fused (AlexNet_layer01.yaml, AlexNet_layer02.yaml, AlexNet_layer02New.yaml, AlexNet_layer03.yaml, AlexNet_layer04.yaml) | - | 37050910000.00 pJ | - |
| **Total** | **59339850000.00 pJ** | **37050910000.00 pJ** | **37.56%** |

## Partition Details

### Stack 0
- **Layers:** AlexNet_layer01.yaml, AlexNet_layer02.yaml, AlexNet_layer02New.yaml, AlexNet_layer03.yaml, AlexNet_layer04.yaml
- **Weight Caching Pattern:** None
- **Tile Sizes:**
  - AlexNet_layer04.yaml: T_in=10, T_out=8
  - AlexNet_layer03.yaml: T_in=12, T_out=10
  - AlexNet_layer02New.yaml: T_in=16, T_out=12
  - AlexNet_layer02.yaml: T_in=20, T_out=16
  - AlexNet_layer01.yaml: T_in=87, T_out=20
- **Spatial Routing:**
  - Layer 0: X=8, Y=0
  - Layer 1: X=8, Y=0
  - Layer 2: X=8, Y=0
  - Layer 3: X=8, Y=0
  - Layer 4: X=8, Y=0

