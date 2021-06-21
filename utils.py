import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as dataset


def image_folder_dataset(root, transform, idx2label):

    old_data = dataset.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dataset.ImageFolder(
        root=root,
        transform=transform,
        target_transform=lambda x: idx2label.index(old_classes[x]),
    )
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def get_list_from_dataloader(dataloader, len=10):
    res = []
    for id, item in enumerate(dataloader):
        res += [item]
        if id >= len - 1:
            return res
