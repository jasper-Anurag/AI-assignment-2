# Q1 Implementation Summary

## Modifications Made

### Q1A: Roulette-Wheel Selection
- **File Modified**: `model_ga.py`
- **Method**: `selection()` (Lines 167-211)
- **Changes**: Replaced tournament selection with roulette-wheel (fitness-proportionate) selection
- **Logging**: Added comprehensive logging of relative fitness scores and probabilities for each generation

### Q1B: Weighted Fitness Function
- **File Modified**: `model_ga.py`
- **Methods**: 
  - `count_conv_fc_params()` (NEW, Lines 66-80)
  - `evaluate_fitness()` (Lines 81-165)
  - `Architecture.__init__()` (Lines 19-32)
- **Changes**: 
  - Separated Conv and FC parameter counting
  - Applied different penalty weights: 0.0008 for Conv, 0.0002 for FC
  - Added logging for parameter counts and penalties

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install torch torchvision numpy
   ```

2. **Run the Code**:
   ```bash
   cd /Users/anurag/Downloads/nas-ga-basic-main
   python nas_run.py
   ```

3. **Logs Location**:
   - Logs will be saved in: `outputs/run_X/nas_run.log`
   - The log file contains:
     - Q1A: Relative fitness scores and probabilities for each generation
     - Q1B: Conv/FC parameter counts and penalties for each architecture evaluation

## Expected Runtime

- **Small Configuration** (population=10, generations=5): ~30-60 minutes on CPU
- **With GPU**: Significantly faster (~5-15 minutes)
- The code uses reduced dataset (5000 train, 1000 validation samples) for faster execution

## Report

See `Q1_Report.md` for detailed documentation including:
- Modified code portions
- Mathematical justifications
- Weight selection rationale


