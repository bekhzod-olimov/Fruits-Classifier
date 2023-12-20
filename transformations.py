# Import library
from torchvision import transforms as T

# # Get transformations based on the input image dimensions
# def get_tfs(im_size = (224, 224), imagenet_normalization = True):
#     mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#     return T.Compose([T.Resize(im_size), T.ToTensor(), T.Normalize(mean = mean, std = std)]) if imagenet_normalization else T.Compose([T.Resize(im_size), T.ToTensor()])

def get_tfs(im_size = 224, crop_size = 256):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return [T.Compose([T.RandomResizedCrop(im_size), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean = mean, std = std)]), T.Compose([T.Resize((crop_size, crop_size)), T.CenterCrop(im_size), T.ToTensor(), T.Normalize(mean = mean, std = std)])]
