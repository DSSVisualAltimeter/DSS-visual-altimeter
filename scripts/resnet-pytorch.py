"""
Purpose: Implement a pretrained model resnet50 to classify altitudes

Progress: Currently this will augment and normalize data for the resnet.

Code taken from Pytorch website

"""

"""
IMPORT LIBRARIES

"""

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim  # set optimizer function
# For Visualizing models must be activated with 'tensorboard --logdir=runs/visual-altimeter-experiment
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt  # For visualizing example images
import time  # analyze time for neural network
import os
import copy



plt.ion()  # interactive mode

"""

DATA LOCATION AND SET UP

Assign the directory where the images are located

Folder format:

/images
├── images/training
│   ├── images/training/60.0 162 images total
│   ├── images/training/75.0 162 images total
│   ├── images/training/90.0 162 images total
├── images/validation
│   ├── images/validation/60.0 41 images total
│   ├── images/validation/75.0 41 images total
│   ├── images/validation/90.0 41 images total

"""
data_dir = '~/projects/DSS-visual-altimeter/images'
print(" This is where the data is located:" , data_dir)

"""
DATA AUGMENTATION, NORMALIZATION, PREPARATION
How does the data come in?
How the images are inputted into the ResNet50 will have a big affect on the outcome

"""
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


# This variable is a list comprehension that assigns both data_transforms to the train and validation file paths
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}

# This variable is a list comprehension that loads in data for the train and validation images
# The data is loaded with these parameters ( batch size=4, shuffle=True, number of workers=4 )
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'validation']}

# Represents the how many images are being processed per folder
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
print("This is how many images the Resnet will be trained on: " ,dataset_sizes)

# Represents the classes
class_names = image_datasets['train'].classes
print("Here are the names of each of the classes", class_names)

# If a computer has CUDA installed then use it, else just use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Shows Images for tensorboard
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# Show example of images (Do not use this function if you already are displaying through tensorboard
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.5)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)


#imshow(out, title="test")
"""

TRAIN MODEL 

"""

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


"""

VISUALIZE MODEL

"""

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {} actual'.format(class_names[preds[j]], [class_names[x] for x in classes]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


""""

FINETUNING

""""

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 10.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 3)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001,
                         momentum=0.9)  # lr = learning rate 0.1-0.001, momentum = 0.5-0.9

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=3)

# visualize_model(model_ft)

# ConvNet as Fixed Feature Extractor

model_conv = torchvision.models.resnet50(pretrained=True)


# Freeze Weights
for param in model_conv.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 3)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=3)

print("Model Convolution final =: ", model_conv)

plt.ioff()
plt.show()

"""
Goal Find a way to display both the predicted and the name of the file.

Problem - Am able to show image and prediction but not name of the file
"""



"""

CONFUSION MATRIX IN COMMAND LINE

"""



nb_classes = 3

confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=torch.float32)
with torch.no_grad():
    for i, (inputs, classes) in enumerate(dataloaders['validation']):
        inputs = inputs.to(device)
        classes = classes.to(device)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for t, p in zip(classes.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)

"""

TENSORBAORD SETUP

"""

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs')

# get some random training images
dataiter = iter(dataloaders['train'])
images, labels = dataiter.next()

# create grid of images
img_grid = torchvision.utils.make_grid(images)

# show images
matplotlib_imshow(img_grid, one_channel=True)

"""

WRITE TO TENSORBOARD

"""
writer.add_image('images', img_grid)
writer.add_graph(model_conv, images)
writer.close()



"""
Sources

Intuition Behind ResNet Architecture

Intuition of Transfer Learning and use of Pretrained Models

Learning to use ResNet in Pytorch (Transfer Learning)
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#sphx-glr-beginner-transfer-learning-tutorial-py
https://pytorch.org/docs/master/notes/autograd.html This site is for understanding freezing your model


Learning tuning Hyper Parameters
https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
https://deeplizard.com/learn/video/ycxulUVoNbk

Learning to visualize and see information about your files
http://ernie55ernie.github.io/machine%20learning/2019/01/06/dogs-and-cats-using-pretrained-convolution-neural-network-for-feature-extraction-and-prediction-with-pytorch.html

Introduction to Tensorboard with Pytorch
https://pytorch.org/docs/stable/tensorboard.html
https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
https://deeplizard.com/learn/video/pSexXMdruFM
https://deeplizard.com/learn/video/ycxulUVoNbk
"""
