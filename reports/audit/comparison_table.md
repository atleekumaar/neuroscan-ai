# NEUROSCAN AI — Stage 4C Comparative Evaluation Table

Evaluated on independent validation subject `BraTS-GLI-00481-000` (dimensions: $240 \times 240 \times 155$, voxel spacing: $1.0\text{ mm}^3$).

### Part A: Raw Model Predictions (Zero Post-Processing, Standard Argmax)

| Model Identifier | Epoch / Stage | WT Dice | WT Prec | WT Sens | WT FP (vox) | TC Dice | ET Dice | Mean Dice | Pred Vol (cm³) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage4B_Epoch2** | NeuroScan_Seg3D_UNet_Stage4B_Epoch2.pth | **`0.1643`** | `0.0898` | `0.9656` | `1,454,624` | `0.0392` | `0.0325` | **`0.0787`** | `1598.1` |
| **Stage4B_Epoch3** | NeuroScan_Seg3D_UNet_Stage4B_Epoch3.pth | **`0.1337`** | `0.0718` | `0.9718` | `1,867,172` | `0.0260` | `0.0269` | **`0.0622`** | `2011.6` |
| **Stage4B_Epoch4_Final** | NeuroScan_Seg3D_UNet_Stage4B_Epoch4_Final.pth | **`0.1513`** | `0.0821` | `0.9668` | `1,607,098` | `0.0316` | `0.0349` | **`0.0726`** | `1750.8` |
| **Stage4A_Pilot** | NeuroScan_Seg3D_UNet_LaptopSafe_Pilot.pth | **`0.1456`** | `0.0790` | `0.9363` | `1,622,829` | `0.0394` | `0.0300` | **`0.0717`** | `1762.0` |
| **Exp1_BalancedPatches** | NeuroScan_Seg3D_UNet_Phase5_Exp1_BalancedPatches.pth | **`0.1198`** | `0.0640` | `0.9298` | `2,020,215` | `0.0425` | `0.0272` | **`0.0632`** | `2158.4` |

*(Ground Truth Volume: WT = 148.597 cm³ | TC = 41.129 cm³ | ET = 31.479 cm³)*

### Part B: Post-Processed Predictions (Intracranial Brain-Mask Constraint Applied)

| Model Identifier | WT Dice (Masked) | $\Delta$ WT Dice | WT FP Masked | FP Reduction (vox) | FP Red % | TC Dice | ET Dice | Mean Dice (Masked) | $\Delta$ Mean Dice |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage4B_Epoch2** | **`0.3014`** | `+0.1371` | `660,053` | `-794,571` | **`-54.6%`** | `0.0767` | `0.0604` | **`0.1462`** | `+0.0675` |
| **Stage4B_Epoch3** | **`0.2635`** | `+0.1298` | `803,047` | `-1,064,125` | **`-57.0%`** | `0.0548` | `0.0498` | **`0.1227`** | `+0.0605` |
| **Stage4B_Epoch4_Final** | **`0.3218`** | `+0.1705` | `600,514` | `-1,006,584` | **`-62.6%`** | `0.0816` | `0.0732` | **`0.1589`** | `+0.0863` |
| **Stage4A_Pilot** | **`0.2951`** | `+0.1495` | `655,216` | `-967,613` | **`-59.6%`** | `0.0810` | `0.0602` | **`0.1454`** | `+0.0737` |
| **Exp1_BalancedPatches** | **`0.2680`** | `+0.1482` | `744,376` | `-1,275,839` | **`-63.1%`** | `0.0801` | `0.0517` | **`0.1333`** | `+0.0701` |
