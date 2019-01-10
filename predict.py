import os
import random
import argparse
import torch
import numpy as np
from PIL import Image

from utils import load_checkpoint, load_category_to_names_mapping

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''    
    im = Image.open(image)
    im = im.resize((256, 256))
    left_upper_dimension = (256 - 224) / 2
    right_lower_dimension = (224 + 256) / 2
    im = im.crop((left_upper_dimension, left_upper_dimension, right_lower_dimension, right_lower_dimension))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std
    im = im.transpose((2,0,1))
    return im

def predict(image_path, model, topk, use_gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if (use_gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    processed_image = process_image(image_path)
    image = torch.from_numpy(np.array([processed_image])).float()
    image = image.to(device)
    output = model.forward(image)
    probabilities = torch.exp(output).data
    top_probabilities = torch.topk(probabilities, topk)[0].tolist()[0]
    top_probable_indices = torch.topk(probabilities, topk)[1].tolist()[0]
    top_probable_classes = [list(model.class_to_idx.keys())[top_probable_indices[i]] for i in range(topk)]
    
    return top_probabilities, top_probable_classes

random_test_dir = random.randint(1, 102)
random_image_filename = random.choice(os.listdir("./flowers/test/{}".format(random_test_dir)))
random_image_path = os.path.join("./flowers/test/{}".format(random_test_dir), random_image_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", dest="checkpoint_file", default="./checkpoint.pth")
    parser.add_argument("--top_k", dest="top_k", default=5)
    parser.add_argument("--image", dest="image_file", default=random_image_path)
    parser.add_argument("--category_names", dest="category_names", default="./cat_to_name.json")
    parser.add_argument("--gpu", action="store_true")
    arguments = parser.parse_args()
    
    model = load_checkpoint(arguments.checkpoint_file)
    category_to_names = load_category_to_names_mapping(arguments.category_names)
    probabilities, classes = predict(image_path=arguments.image_file, model=model, topk=arguments.top_k, use_gpu=arguments.gpu)
    probable_label = classes[probabilities.index(max(probabilities))]
    probable_label_index = list(category_to_names.keys())[int(probable_label)]
    print("probabilities={} classes={}\n".format(probabilities, classes))
    print("Flower in location {} is {:.0%} likely to be '{}'.".format(arguments.image_file, max(probabilities), category_to_names.get(probable_label_index).title()))