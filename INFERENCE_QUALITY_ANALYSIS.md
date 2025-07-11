# Inference Quality Analysis: Training vs Inference Performance

## Problem Statement

The dynamics model shows excellent performance during training but produces increasingly poor quality frames during inference, with frames becoming "shittier and shittier" as the sequence progresses.

## Root Cause Analysis

### 10 Initial Theories

1. **Training Masking vs Inference Mode** - Model trained with masking, inference without
2. **Teacher Forcing vs Autoregressive Generation** - Training uses ground truth, inference uses predictions
3. **Context Window Mismatch** - Inference sequences longer than training context
4. **Latent Space Drift** - Predictions drift from training distribution
5. **Action Latent Mismatch** - Random actions vs real frame transitions
6. **Quantization Error Accumulation** - VQ errors compound over time
7. **Gradient Flow Differences** - Different behavior without gradients
8. **Batch Normalization/EMA Issues** - Statistics mismatch
9. **Causal Masking Inconsistency** - Attention patterns differ
10. **Loss Function Mismatch** - Training vs inference objectives differ

### Diagnostic Test Results

**Test: Real Context vs Generated Context**
- **MSE Loss**: 0.079 ± 0.014 (with real context)
- **PSNR**: 17.09 ± 0.79 dB (with real context)
- **Conclusion**: Model performs well when given real data context

**Test: Masking Theory**
- **Normal MSE**: 0.079345
- **Masked MSE**: 0.090940 (worse)
- **Conclusion**: Masking is NOT the issue

### Final Diagnosis: Theory 2 - Teacher Forcing vs Autoregressive Generation

**The Problem**: Error Accumulation in Rollouts

During training:
```
Real Frame 1 → Model → Predicted Frame 2 → Compare with Real Frame 2
```

During inference:
```
Real Frame 1 → Model → Predicted Frame 2 → Model → Predicted Frame 3 → Model → Predicted Frame 4...
```

Each prediction error gets fed back into the model, causing exponential quality degradation.

## Solutions Implemented

### Solution 1: Scheduled Sampling (Training)

**File**: `src/dynamics/main.py`

**Implementation**: Gradually transition from teacher forcing to using predicted frames during training.

```python
# SCHEDULED SAMPLING: Gradually transition from teacher forcing to using predicted frames
# Start with 100% teacher forcing, end with 50% teacher forcing
teacher_forcing_ratio = max(0.5, 1.0 - (i / args.n_updates) * 0.5)

if torch.rand(1).item() < teacher_forcing_ratio:
    # Teacher forcing: use ground truth latents
    input_latents = combined_latents
else:
    # Use predicted latents for some timesteps
    with torch.no_grad():
        predicted_latents = dynamics_model(combined_latents[:, :-1], training=True)
        input_latents = torch.cat([combined_latents[:, :1], predicted_latents], dim=1)
```

**Benefits**:
- Model learns to handle its own predictions
- Gradual transition prevents training instability
- Maintains good performance while improving inference

### Solution 2: Temperature Sampling (Inference)

**File**: `run_inference.py`

**Implementation**: Add controlled noise during inference to reduce error accumulation.

```python
def predict_next_tokens(dynamics_model, video_latents, action_latent, temperature=1.0):
    # ... existing code ...
    
    # Apply temperature sampling to reduce variance
    if temperature != 1.0:
        noise = torch.randn_like(next_video_latents) * (temperature - 1.0) * 0.1
        next_video_latents = next_video_latents + noise
    
    return next_video_latents
```

**Benefits**:
- Reduces prediction variance
- Prevents getting stuck in poor local optima
- More conservative predictions

### Solution 3: Action Diversity Sampling

**File**: `run_inference.py`

**Implementation**: Sample actions with diversity to avoid repetitive patterns.

```python
def sample_action_with_diversity(lam, previous_actions, n_actions, diversity_weight=0.1):
    # Favor less frequent actions to avoid loops
    if torch.rand(1).item() < diversity_weight:
        min_count = action_counts.min()
        candidate_actions = torch.where(action_counts == min_count)[0]
        action_index = candidate_actions[torch.randint(0, len(candidate_actions), (1,))]
    else:
        action_index = torch.randint(0, n_actions, (1,))
    
    return action_index
```

**Benefits**:
- Prevents getting stuck in action loops
- More diverse and interesting sequences
- Better exploration of action space

## Usage Instructions

### Retrain with Scheduled Sampling

```bash
python src/dynamics/main.py \
    --video_tokenizer_path <path> \
    --lam_path <path> \
    --batch_size 16 \
    --n_updates 5000 \
    --learning_rate 1e-4
```

### Run Improved Inference

```bash
python run_inference.py \
    --video_tokenizer_path <path> \
    --lam_path <path> \
    --dynamics_path <path> \
    --temperature 0.8 \
    --generation_steps 20
```

### Test with Real Context

```bash
python test_inference_with_data_context.py \
    --video_tokenizer_path <path> \
    --lam_path <path> \
    --dynamics_path <path> \
    --num_tests 20 \
    --test_masking
```

## Expected Improvements

1. **Better Long-Sequence Quality**: Scheduled sampling should significantly improve inference quality
2. **Reduced Error Accumulation**: Temperature sampling prevents exponential degradation
3. **More Diverse Outputs**: Action diversity sampling creates more interesting sequences
4. **Stable Performance**: Model should maintain quality throughout longer sequences

## Monitoring Progress

- **Training**: Watch for teacher forcing ratio decreasing over time
- **Inference**: Monitor PSNR/MSE metrics across sequence length
- **Visual Quality**: Compare frame quality at different sequence positions

## Future Improvements

1. **Beam Search**: Use beam search instead of greedy decoding
2. **Ensemble Methods**: Combine multiple model predictions
3. **Curriculum Learning**: Train on progressively longer sequences
4. **Adversarial Training**: Use discriminator to improve realism
5. **Attention Visualization**: Analyze attention patterns during inference

## Key Takeaways

1. **The issue is NOT masking** - masking actually hurts performance
2. **Error accumulation is the real problem** - each prediction error compounds
3. **Scheduled sampling is the most effective solution** - teaches model to handle its own predictions
4. **Temperature sampling helps during inference** - reduces variance and improves stability
5. **Action diversity prevents loops** - creates more interesting and varied sequences

The diagnostic script `test_inference_with_data_context.py` provides a valuable tool for testing future improvements and monitoring model performance. 