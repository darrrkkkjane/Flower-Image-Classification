import torch
from torch import optim 
from torchvision import models, datasets, transforms

def load_checkpoint(filepath):
    """
    Function to load the checkpoint.
    
    filepath: image file from where the checkpoint dict is loaded
    """
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained = True)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model