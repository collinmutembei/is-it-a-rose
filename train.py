import argparse
import torch
from collections import OrderedDict
from torch import cuda, nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from utils import save_checkpoint

def train(model, criterion, optimizer, dataloaders, epochs, use_gpu):
    steps = 0
    running_loss = 0
    accuracy = 0
    phases = OrderedDict([
        ("training_phase", model.train),
        ("validation_phase", model.eval)
    ])
    device = torch.device("cuda:0" if (use_gpu and cuda.is_available()) else "cpu")
    model.to(device)
    print('Training initialized\n')
    for epoch in range(epochs):
        for phase, phase_operation in phases.items():
            phase_operation()
            pass_count = 0
            for inputs, labels in dataloaders.get(phase[:5]):
                pass_count += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)
                if phase[:5] == "train":
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                probabilities = torch.exp(output).data
                equality = (labels.data == probabilities.max(1)[1])
                device_specific_float_type = torch.cuda.FloatTensor() if cuda.is_available() else torch.FloatTensor()
                accuracy = equality.type_as(device_specific_float_type).mean()
                
            if phase[:5] == "train":
                print("Epoch: {}/{} ".format(epoch + 1, epochs))
                print("Training Loss: {:.4f}  ".format(running_loss / pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss / pass_count), "Accuracy: {:.4f}\n".format(accuracy))

            running_loss = 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier neural network")
    parser.add_argument("--data_dir", dest="data_directory")
    parser.add_argument("--save_dir", dest="save_directory")
    parser.add_argument("--arch", dest="arch", default="vgg16", choices=["vgg16", "vgg19"])
    parser.add_argument("--learning_rate", dest="learning_rate", default=0.01)
    parser.add_argument("--hidden_units", dest="hidden_units", default=1024)
    parser.add_argument("--epochs", dest="epochs", default=4)
    parser.add_argument("--gpu", action="store_true")
    arguments = parser.parse_args()
    
    data_dir = arguments.data_directory
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"
    
    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomRotation(60),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "valid": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])     
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {
        "train": datasets.ImageFolder(root=train_dir, transform=data_transforms.get("train")),
        "valid": datasets.ImageFolder(root=valid_dir, transform=data_transforms.get("valid")),
        "test": datasets.ImageFolder(root=test_dir, transform=data_transforms.get("test"))
    }

    dataloaders = {
        "train": data.DataLoader(image_datasets.get("train"), batch_size=30, shuffle=True),
        "valid": data.DataLoader(image_datasets.get("valid"), batch_size=6, shuffle=True),
        "test": data.DataLoader(image_datasets.get("test"), batch_size=6, shuffle=True)
    }
    
    model = getattr(models, arguments.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    if arguments.arch in ["vgg16", "vgg19"]:
        input_features_count = list(model.classifier.children())[0].in_features
        classifier = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(input_features_count, int(arguments.hidden_units))),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(p=0.5)),
                ("fc2", nn.Linear(int(arguments.hidden_units), 102)),
                ("output", nn.LogSoftmax(dim=1))
            ]))
    
    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=float(arguments.learning_rate))
    epochs = int(arguments.epochs)
    use_gpu = arguments.gpu
    train(model, criterion, optimizer, dataloaders, epochs, use_gpu)
    model.class_to_idx = image_datasets.get("train").class_to_idx
    save_checkpoint(model, optimizer, arguments, classifier, arguments.save_directory)