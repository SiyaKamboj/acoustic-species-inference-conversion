# export_to_onnx.py

import torch
#from model_wrapper import get_wrapped_model
from model_wrapper import wrapped_model

# Get the model
# wrapped_model = get_wrapped_model(num_classes=21)  # adjust class count
wrapped_model.eval()

# Create dummy input
dummy_input = torch.randn(1, 1, 256, 431)  # [B, C, H, W]

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "efficientnet_b1_export.onnx",
    input_names=["audio_in"],
    output_names=["logits"],
    dynamic_axes={
        "audio_in": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=17  # adjust if needed
)

print("Model exported to efficientnet_b1_export.onnx")
