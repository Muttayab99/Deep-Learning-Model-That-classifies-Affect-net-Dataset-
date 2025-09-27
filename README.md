# Deep-Learning-Model-That-classifies-Affect-net-Dataset-

# Facial Expression Recognition Using CNN Baselines

## 1. Overview
This project evaluates **baseline Convolutional Neural Network (CNN) models** for multi-task **facial expression recognition** on the *Affect-Facial-Expressions-Dataset*.  

- **Tasks:**
  - **Classification:** 8 categories based on AffectNet labeling  
    *(0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger, 7: Contempt)*  
  - **Regression:** Continuous **valence** (pleasant–unpleasant) and **arousal** (activated–deactivated) values in the range `[-1, 1]`.

- **Baselines:**
  - **ResNet18** (simple, residual connections, stable benchmark).  
  - **EfficientNet-B0** (efficient, compound scaling, mobile-friendly).  
  - **Ensemble** (0.6 ResNet + 0.4 EfficientNet with Test-Time Augmentation).  

- **Techniques:**
  - Transfer learning with ImageNet weights.  
  - Data augmentation: RandAugment, flips, rotations, jitter, crops.  
  - Regularization: Dropout, Mixup, CutMix.  
  - Multi-task weighted losses (classification + regression).  
  - Evaluation with categorical metrics (Accuracy, F1) and continuous metrics (RMSE, CORR, SAGR, CCC).  

All experiments were run in a **GPU environment**, with reproducibility ensured via seeded random states.

---

## 2. Network Details

### ResNet18
- Backbone: residual network (18 layers).
- Modified for multi-task learning:
  - **Classification head:**  
    `Dropout(0.5) → Linear(512→512) → ReLU → Dropout(0.5) → Linear(512→8)`
  - **Regression head:**  
    `Dropout(0.5) → Linear(512→512) → ReLU → Dropout(0.5) → Linear(512→2) → Tanh`

### EfficientNet-B0
- Backbone: mobile inverted bottlenecks with compound scaling.
- Similar modifications:
  - Classifier replaced with identity (extracting 1280 features).
  - Same multi-task heads as ResNet18.

### Ensemble
- Weighted average of predictions (**0.6 ResNet + 0.4 EfficientNet**).
- **Test-Time Augmentation (TTA):** 8 augmentations (flips, ±5°/±10° rotations).

### Parameters
- ResNet18: ~512K (heads only), ~11.7M (full).  
- EfficientNet-B0: ~512K (heads only), ~5.3M (full).  
- Input: `224x224 RGB`.  
- Output: softmax probabilities (8 classes), and valence/arousal in `[-1, 1]`.


## 3. Training Settings
- **Optimizer:** AdamW, weight decay `1e-4`, LR `1e-3` (OneCycleLR scheduler).  
- **Loss Functions:**
  - Classification: Focal Loss (γ=2.0).  
  - Regression: Smooth L1 Loss.  
  - Combined: `0.9 * classification + 0.1 * regression`.  
- **Batch size:** 32.  
- **Epochs:** up to 50 (early stopping with patience=7).  
- **Score for early stopping:** `0.7 * Accuracy + 0.3 * CCC_val`.  
- **Regularization:** Dropout=0.5, Mixup (α=0.2), CutMix (α=1.0).  
- **Augmentation:** RandAugment, flips, rotations (15°), jitter, crops.  
- **Transfer learning:**  
  - Backbones frozen initially.  
  - Partial unfreeze at epoch 5 (last layers).  
  - Full unfreeze at epoch 15.  
- **Other:** Mixed precision training (AMP), TTA during evaluation.


## 4. Dataset Splits
- **Dataset:** Affect-Facial-Expressions (~4K images).  
- **Annotations:** expression class (0–7), valence/arousal ([-1, 1]).  
- **Splits:** 80% train (3,200), 20% validation (800). Stratified by class.  
- **Preprocessing:**  
  - Train: heavy augmentation.  
  - Val: resize → center crop → normalize (ImageNet stats).  
- **Handling invalid labels:** valence/arousal = -2 replaced with 0.0.  


## 5. Transfer Learning
- Pretrained weights:  
  ```python
  models.resnet18(weights=ResNet18_Weights.DEFAULT)
  models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)


## 6. Training Graphs
*(Graphs attached in repo as images)*  

- **ResNet18:**  
  - Training loss decreases from ~1.4 → ~1.0.  
  - Validation loss stabilizes around ~1.2 (minor overfitting after epoch 30).  
  - Train accuracy increases to ~0.8; validation accuracy plateaus at ~0.48 (epoch ~20).  

- **EfficientNet-B0:**  
  - Training loss: ~1.6 → ~1.1.  
  - Validation loss fluctuates between ~1.3–1.4 (sensitive to augmentations).  
  - Train accuracy rises to ~0.7; validation accuracy ~0.42 (stable but lower than ResNet).  

- **Observation:** Both converge, early stopping prevents severe overfitting.


## 7. Performance Measures
| Model            | Accuracy | F1-Score | Valence RMSE | Arousal RMSE | Val CCC | Aro CCC |
|------------------|----------|----------|--------------|--------------|---------|---------|
| ResNet18         | ~0.48    | ~0.46    | ~0.28        | ~0.32        | 0.42    | 0.38    |
| EfficientNet-B0  | ~0.42    | ~0.40    | ~0.31        | ~0.34        | 0.39    | 0.35    |
| Ensemble (Res+Ef)| ~0.50    | ~0.48    | ~0.27        | ~0.31        | 0.45    | 0.40    |

*Metrics are reported on the validation split; ensemble improves robustness.*


**Patterns:**  
- Frequent confusion between **neutral/sad** and other subtle emotions.  
- Ensemble reduces but does not eliminate errors.  
- Dataset challenges: variations in age, pose, occlusion, illumination.


## 9. Rationale for Baselines
- **ResNet18:** lightweight benchmark with residual connections → stable learning.  
- **EfficientNet-B0:** FLOPs-optimized, mobile-friendly, efficient compound scaling.  
- **Ensemble:** leverages complementary strengths, improves generalization and robustness.  


## 10. Conclusion
- **ResNet18** achieved the best standalone categorical accuracy (~48%).  
- **EfficientNet-B0** performed slightly lower (~42%) but remains efficient.  
- **Ensemble** boosted robustness and overall performance (~50% accuracy, higher CCC).  
- Main difficulty: **subtle expression confusion** (Neutral vs Sad, Anger vs Contempt).  
  
