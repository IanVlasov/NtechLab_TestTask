import os
import json
import sys
import torch
import numpy as np

from torchvision import datasets, transforms
from torch.utils import data
from src.nn_structure import Net

def make_predictions(path):
     """Function makes predictions and saves them in file
     'process_results.json' in the given {path}
     Fotos should be stored inside additional folder in the
     given path (e.g. "path/data")
     
     Input
     -------
     path: path where the file with results should be saved
              Fotos should be stored inside additional folder in the
              given path (e.g. "path/data")
     """
    
    cur_dir = os.getcwd()
    output_dict = {}
    
    classes = {0: 'female',
                     1: 'male'}
    
    model = Net()
    model.load_state_dict(torch.load('checkpoint.pt'))
    
    transform = transforms.Compose([
                                transforms.Resize((64, 64)),
                                transforms.Grayscale(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), 
                                                     (0.5,))
    ])
    
    dataset = datasets.ImageFolder(root = path, 
                                   transform = transform)
    
    loader = data.DataLoader(dataset, batch_size = 1, shuffle = False)
    
    
    with torch.no_grad():
        for i, (image, label) in enumerate(loader, 0):
            output = model(image)
            prediction = (np.squeeze(torch.sigmoid(output.data)) > 0.5) * 1
            sample_path, _ = loader.dataset.samples[i]
            sample_fname = os.path.basename(sample_path)
            output_dict[sample_fname] = classes[prediction.item()]
    
    output_fname = os.path.join(path, 'process_results.json')
    
    with open(output_fname, 'w') as fp:
        json.dump(output_dict, fp)
        
if __name__ == "__main__":
	make_predictions(sys.argv[1])