import numpy as np
from torchvision import models, datasets, transforms 
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    # Processing a PIL image for use in a PyTorch model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])

    img_pil = Image.open(image)
    img = trans(img_pil)
     
#     image = np.array(img)
    
#     image = image/255.

#     image = (image-mean)/std
    
#     image = image.transpose((2, 0, 1))
    
    return img