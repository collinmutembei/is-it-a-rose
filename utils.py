import json

def save_checkpoint(model, optimizer, args, classifier):
    
    checkpoint = {
        'arch': args.arch, 
        'model': model,
        'learning_rate': args.learning_rate,
        'hidden_units': args.hidden_units,
        'classifier' : classifier,
        'epochs': args.epochs,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }

    torch.save(checkpoint, 'checkpoint.pth')

    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint.get("model")
    model.classifier = checkpoint.get("classifier")
    learning_rate = checkpoint.get("learning_rate")
    epochs = checkpoint.get("epochs")
    optimizer = checkpoint.get("optimizer")
    model.load_state_dict(checkpoint.get("state_dict"))
    model.class_to_idx = checkpoint.get("class_to_idx")
    
    return model


def load_category_to_names_mapping(category_file):
    with open(category_file, "r") as f:
        category_to_names = json.load(f)
    return category_to_names
