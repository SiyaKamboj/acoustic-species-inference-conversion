import torch
from pyha_analyzer.models import EfficentNet  

# Instantiate your model as usual
model = EfficentNet(num_classes=21) #21 bird classes
model.eval()  # set to evaluation mode

# Define a wrapper with only the parts needed for inference
class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model  # Access the internal HuggingFace model

    def forward(self, x):
        return self.model(pixel_values=x).logits  # Return only logits (no loss, no labels)
    
    def get_wrapped_model():
        return wrapped_model


wrapped_model = ExportWrapper(model)
wrapped_model.eval()
