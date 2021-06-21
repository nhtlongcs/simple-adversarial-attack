import torchvision.utils
from utils import imshow
from tqdm.notebook import tqdm

import torch

from utils import idx2label, device


def eval(
    model, viz=False, dataset=None, loader=None, verbose=False, batch_size=1,
):
    print("True Image & Predicted Label")

    model.eval()

    correct = 0
    total = 0
    pbar = tqdm(loader, total=len(dataset) // batch_size) if verbose else loader
    with torch.no_grad():
        for images, labels in pbar:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, pre = torch.max(outputs.data, 1)

            total += 1
            correct += (pre == labels).sum()
            if viz:
                imshow(
                    torchvision.utils.make_grid(images.cpu().data, normalize=True),
                    [idx2label[i] for i in pre],
                )

    print("Accuracy of test text: %f %%" % (100 * float(correct) / total / batch_size))

