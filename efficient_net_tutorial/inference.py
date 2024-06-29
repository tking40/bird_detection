import torch
# import cv2
import numpy as np
import torch.nn as nn
import glob as glob
import os
from model import build_model
from datasets import get_datasets, get_data_loaders, get_valid_transform, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from train import validate
# Constants.
DATA_PATH = '../data/birds/test'
IMAGE_SIZE = 224
DEVICE = 'mps'


if __name__ == '__main__':
    pretrained = False
    # dataset_train, dataset_valid, dataset_classes = get_datasets(pretrained=pretrained)
    # _, dataloader_valid = get_data_loaders(dataset_train, dataset_valid)

    dataset = datasets.ImageFolder(
        DATA_PATH, 
        transform=(get_valid_transform(IMAGE_SIZE, pretrained))
    )
    dataset_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )

    # print(dataset.classes)
    # exit()

    # Class names.
    class_names = dataset.classes
    # Load the trained model.
    device = torch.device(DEVICE)
    model = build_model(pretrained=True, fine_tune=False, num_classes=len(class_names)).to(device)
    checkpoint = torch.load('./outputs/model_pretrained_True.pth', map_location=DEVICE)
    print('Loading trained model weights...')
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()

    valid_epoch_loss, valid_epoch_acc = validate(model, dataset_loader, criterion, device)
    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")


    # Get all the test image paths.
    # all_image_paths = glob.glob(f"{DATA_PATH}/*")
    # Iterate over all the images and do forward pass.
    # for image_path in all_image_paths:

    # for i, data in enumerate(dataset_loader):
    #     # Get the ground truth class name from the image path.
    #     with torch.no_grad():
    #         image, label = data
    #         print(class_names[label.item()])

    #         # Display the image
    #         plt.imshow(image[0].permute(1, 2, 0))
    #         plt.title(class_names[label.item()])
    #         plt.show()

    #         if i == 0:
    #             break

    # gt_class_name = image_path.split(os.path.sep)[-1].split('.')[0]
    # gt_class_name = labels
    # # Read the image and create a copy.
    # image = cv2.imread(image_path)
    # orig_image = image.copy()
    
    # # Preprocess the image
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    # image = transform(image)
    # image = torch.unsqueeze(image, 0)
    # image = image.to(DEVICE)
    
    # # Forward pass throught the image.
    # outputs = model(image)
    # outputs = outputs.detach().numpy()
    # pred_class_name = class_names[np.argmax(outputs[0])]
    # print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
    # # Annotate the image with ground truth.
    # cv2.putText(
    #     orig_image, f"GT: {gt_class_name}",
    #     (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
    #     1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
    # )
    # # Annotate the image with prediction.
    # cv2.putText(
    #     orig_image, f"Pred: {pred_class_name.lower()}",
    #     (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
    #     1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
    # ) 
    # cv2.imshow('Result', orig_image)
    # cv2.waitKey(0)
    # cv2.imwrite(f"./outputs/{gt_class_name}.png", orig_image)