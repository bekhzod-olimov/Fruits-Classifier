# Import libraries
import torch, torchvision, os
from torch.utils.data import random_split, Dataset, DataLoader
from torch import nn; from torchvision import transforms as T
from PIL import Image; from glob import glob
# Set manual seed for reproducibility
torch.manual_seed(2023)

class CustomDataset(Dataset):

    """

    This class get several arguments and returns dataset to be trained.

    Parameters:

    root            - path to the data, str;
    data            - data name, str;
    transformations - transformations to be applied, torchvision transforms object.
    
    """
    
    def __init__(self, root, data, transformations = None):
        
        self.transformations = transformations
        # Get image paths from the given directory
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/{data}/*/*"))]
        
        self.cls_names, self.cls_counts, count, data_count = {}, {}, 0, 0
        # Go through every image path and get class names along with class counts
        for idx, im_path in enumerate(self.im_paths):
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names: self.cls_names[class_name] = count; self.cls_counts[class_name] = 1; count += 1
            else: self.cls_counts[class_name] += 1
        
    # Function to get class name
    def get_class(self, path): return os.path.dirname(path).split("/")[-1]
    
    # Function to get number of data
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):

        """

        This function gets index and returns corresponding data pair.

        Parameter:

        idx    - index, int.

        Outputs:

        im     - image, tensor;
        gt     - ground truth (label), int.
        
        """
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = self.cls_names[self.get_class(im_path)]
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt
    
def get_dls(root, transformations, bs, split = [0.9, 0.05], ns = 4):

    """

    This function gets several parameters and returns dataloaders and a class names file.

    Parameters:

        root             - path to data, str;
        transformations  - transformations to be applied, transforms object;
        bs               - mini batch size, int;
        split            - split ratio, list -> float;
        ns               - number of workers, int.  

    Outputs:

        tr_dl            - train dataloader, torch dataloader object;
        val_dl           - validation dataloader, torch dataloader object;
        ts_dl            - test dataloader, torch dataloader object;
        cls_names        - class names of the dataset, dict
    
    """
        
    # Get datasets
    tr_ds = CustomDataset(root = root, data = "train", transformations = transformations)
    vl_ds = CustomDataset(root = root, data = "val", transformations = transformations)
    ts_ds = CustomDataset(root = root, data = "test", transformations = transformations)

    # Get dataloaders
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(vl_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, tr_ds.cls_names
