import os
import struct
import torch
import numpy as np
from transformers import AutoModel

def export_model(model_id="sentence-transformers/all-MiniLM-L6-v2", output_path="models/minilm_weights.bin"):
    print(f"Loading {model_id} from HuggingFace...")
    # Load the pre-trained MiniLM model
    model = AutoModel.from_pretrained(model_id)
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Exporting raw weights to {output_path}...")
    with open(output_path, "wb") as f:
        # Iterate over all the parameters (weights/biases) in the network
        for name, param in model.named_parameters():
            # Convert PyTorch tensor to raw numpy array in float32 format
            tensor = param.detach().cpu().numpy().astype('float32')
            
            # Quantize only 2D weights (Linear layers), leave biases/layernorm as float32
            is_quantizable = len(tensor.shape) == 2
            
            # Write the length of the layer name, then the name itself
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write the number of dimensions of the tensor
            f.write(struct.pack('I', len(tensor.shape)))
            
            # Write the actual shape (e.g., 384 x 384)
            for dim in tensor.shape:
                f.write(struct.pack('I', dim))
                
            if is_quantizable:
                # DType = 1 (INT8)
                f.write(struct.pack('I', 1))
                
                # Symmetric per-tensor quantization
                max_val = np.max(np.abs(tensor))
                scale = 127.0 / max_val if max_val != 0 else 1.0
                f.write(struct.pack('f', scale))
                
                # Quantize and write
                q_tensor = np.round(tensor * scale).astype(np.int8)
                f.write(q_tensor.tobytes())
                print(f"Exported: {name:50} | Shape: {tensor.shape} | INT8 (Scale: {scale:.4f})")
            else:
                # DType = 0 (F32)
                f.write(struct.pack('I', 0))
                
                # Write the raw float32 data
                f.write(tensor.tobytes())
                print(f"Exported: {name:50} | Shape: {tensor.shape} | F32")
            
    print("\nExport complete! The C code can now read this binary file.")

if __name__ == "__main__":
    export_model()
