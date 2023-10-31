# PROGRAMMER: Janeth Manyalla
# DATE CREATED:  26/03/2023


import torch
import json
from torch import nn
from torch import optim 
import torch.nn.functional as F
from get_input_args import get_input_args
from torchvision import models, datasets, transforms

# Example call:
#     python train.py flowers

def main():
    
    in_arg = get_input_args()
    
    data_dir = in_arg.data_directory

    # Loading the data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Defining  transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_val_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)

    val_data = datasets.ImageFolder(valid_dir, transform=test_val_transforms)

    test_data = datasets.ImageFolder(test_dir, transform=test_val_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(val_data, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    # Label mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Loading the model
    # Loading the VGG16 model
    model = models.vgg16(pretrained = True)


    # Freezing parameters
    for param in model.parameters():
        param.requires_grad=False


    from collections import OrderedDict    
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, 4096)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(4096, 1024)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(1024, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    # Defining optimizer and Loss function
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    epochs = 15

    model.to(device)

    # Looping through epochs, calculating loss and accuracy for training and validation
    for e in range(epochs):
        train_loss, val_loss, train_acc, val_acc = 0., 0., 0., 0.
        model.train()
        for image, label in trainloader:
            optimizer.zero_grad()

            image = image.to(device)
            label = label.to(device)

            output = model.forward(image)
            loss = criterion(output, label)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

        model.eval()
        with torch.no_grad():
            for image, label in validloader:
                image = image.to(device)
                label = label.to(device)
                output = model(image)
                loss = criterion(output, label)
                val_loss += loss.item()

                # Calculate accuracy
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class==label.view(*top_class.shape)
                val_acc += torch.mean(equals.type(torch.FloatTensor)).item()

        # Taking averages
        train_loss /= len(trainloader)
        val_loss /= len(validloader)
        val_acc /= len(validloader)

        print("Train Loss: {:.3f}.. ".format(train_loss),
              "Validation Loss: {:.3f}.. ".format(val_loss),
              "Validation Acc: {:.3f}.. ".format(val_acc))
        
    images = datasets.ImageFolder(test_dir)
    model.class_to_idx = images.class_to_idx 
    
    # Save the checkpoint 
    checkpoint = {'epochs': epochs,
                  'classifier': model.classifier,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict(),
                  'model_state_dict': model.state_dict()}


    torch.save(checkpoint, 'checkpoint.pth')
        
        
# Call to main function to run the program
if __name__ == "__main__":
    main()
    