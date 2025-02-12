import torch
import bitsandbytes as bnb

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    try:
        # Try a simple bitsandbytes operation (just to test)
        # For example, create an 8-bit optimizer:
        optimizer = bnb.optim.Adam(torch.randn(10, 10), lr=0.001)
        print("bitsandbytes initialized successfully!")
    except Exception as e:
        print(f"bitsandbytes initialization error: {e}")
else:
    print("CUDA is not available. Please check your CUDA installation.")
    