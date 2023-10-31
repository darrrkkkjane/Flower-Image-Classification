import torch
from process_image import process_image
from get_input_args_predict import get_input_args


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_arg = get_input_args()
    device = in_arg.gpu
    model.eval()
    # Get the class probabilities
    model = model.to(device)
    
    image = torch.tensor(process_image(image_path))
    image = image.to(device)
    
    # resize the tensor (add dimension for batch)
    image = image.unsqueeze_(0)
    with torch.no_grad():
        ps = torch.exp(model(image))

    top_p, top_class = ps.topk(5, dim=1)
   
    return top_p, top_class