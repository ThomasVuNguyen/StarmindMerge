# GTX 1050 Ti Unsloth Setup Guide

Successfully running Unsloth with Gemma3 270M on NVIDIA GeForce GTX 1050 Ti (CUDA Capability 6.1)

## Prerequisites

- NVIDIA GeForce GTX 1050 Ti
- Python 3.12.3
- CUDA 6.1 support
- Linux environment

## Working Configuration

### Key Package Versions

The following combination has been **tested and confirmed working**:

```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
unsloth==2025.9.1
unsloth_zoo==2025.9.2
transformers==4.56.1
numpy==2.3.2
bitsandbytes==0.47.0
triton==3.3.1
```

### Installation Steps

1. **Create Python Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch with CUDA 11.8 Support**
   ```bash
   pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Unsloth**
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   ```

4. **Verify Installation**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU name:', torch.cuda.get_device_name(0))"
   ```

## Test Script Execution

Run the Gemma3 inference script with:

```bash
source venv/bin/activate
TORCH_COMPILE_DISABLE=1 python gemma_inference.py
```

### Expected Output

```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
Loading Gemma3 270M model...
==((====))==  Unsloth 2025.9.1: Fast Gemma3 patching. Transformers: 4.56.1.
   \\   /|    NVIDIA GeForce GTX 1050 Ti. Num GPUs = 1. Max memory: 3.937 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.7.1+cu118. CUDA: 6.1. CUDA Toolkit: 11.8. Triton: 3.3.1
\        /    Bfloat16 = FALSE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
```

## Configuration Details

### System Configuration
- **GPU**: NVIDIA GeForce GTX 1050 Ti
- **CUDA Version**: 6.1
- **CUDA Toolkit**: 11.8
- **Available GPU Memory**: 3.937 GB
- **Precision**: float32 (bfloat16 disabled for GTX 1050 Ti)
- **Flash Attention**: Disabled (xFormers = None, FA2 = False)

### Performance Notes
- Model runs in **float32** precision (bfloat16 not supported on GTX 1050 Ti)
- Uses **16bit LoRA** for memory efficiency
- Flash Attention is disabled due to hardware limitations
- xFormers warnings can be ignored as they don't affect functionality

## Troubleshooting

### Common Issues

1. **CUDA Capability Warnings**
   - GTX 1050 Ti (capability 6.1) generates warnings but still works
   - Warnings about minimum capability 7.0 can be ignored for this configuration

2. **Memory Limitations**
   - 4GB VRAM limits model size and batch size
   - Use smaller models or increase system swap if needed

3. **xFormers Warnings**
   - Version mismatch warnings are expected and don't affect functionality
   - Can be suppressed by setting `XFORMERS_MORE_DETAILS=0`

### Environment Variables

```bash
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_INDUCTOR=0
export TORCHINDUCTOR_MAX_AUTOTUNE=0
export TORCH_COMPILE_DISABLE=1
```

## Full Requirements.txt

For exact reproduction, use the complete requirements file:

```
torch==2.7.1+cu118
torchvision==0.22.1+cu118
unsloth==2025.9.1
unsloth_zoo==2025.9.2
transformers==4.56.1
accelerate==1.10.1
bitsandbytes==0.47.0
datasets==3.6.0
peft==0.17.1
trl==0.22.2
numpy==2.3.2
triton==3.3.1
# ... (see working_requirements.txt for complete list)
```

## Success Confirmation

✅ **Model Loading**: Successfully loads Gemma3 270M model  
✅ **GPU Detection**: Properly detects and utilizes GTX 1050 Ti  
✅ **Inference**: Generates coherent text responses  
✅ **Memory Management**: Efficient memory usage within 4GB VRAM limits  
✅ **Replication Verified**: Successfully replicated from scratch in myenv environment

### Replication Test Results

The setup has been **successfully replicated** using the following process:

1. **Fresh Environment**: Completely removed and recreated `myenv`
2. **Exact Package Installation**: Installed PyTorch 2.7.1+cu118 and Unsloth
3. **Working Verification**: Model loaded and generated responses successfully

```bash
# Successful test output:
✅ Model loaded successfully!
✅ Response: ```python
def hello_world():
  """A simple function to print a greeting."""
  print("Hello, world!")
```
🎉 SUCCESS: GTX 1050 Ti Unsloth setup is working!
```  

## Note on Compatibility

While Unsloth officially requires CUDA capability 7.0+, this configuration demonstrates that GTX 1050 Ti (capability 6.1) can work with:
- Proper PyTorch version (2.7.1+cu118)
- Disabled flash attention
- Float32 precision
- Appropriate environment variables

This setup provides a working solution for older Pascal architecture GPUs despite official compatibility limitations.