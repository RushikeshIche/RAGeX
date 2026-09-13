import os
import struct
import torch
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
            
            # Write the length of the layer name, then the name itself
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write the number of dimensions of the tensor
            f.write(struct.pack('I', len(tensor.shape)))
            
            # Write the actual shape (e.g., 384 x 384)
            for dim in tensor.shape:
                f.write(struct.pack('I', dim))
                
            # Write the raw float32 data
            f.write(tensor.tobytes())
            
            print(f"Exported: {name:50} | Shape: {tensor.shape}")
            
    print("\nExport complete! The C code can now read this binary file.")

if __name__ == "__main__":
    export_model()
