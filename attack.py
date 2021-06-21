import torch
import torch.nn as nn
import torchvision.utils

from tqdm.notebook import tqdm

from fgsm import fgsm_attack
from utils import idx2label, imshow, device

loss = nn.CrossEntropyLoss()


def attack(
    model,
    viz=False,
    dataset=None,
    loader=None,
    batch_size=1,
    verbose=False,
    attack_times=1000,
    eps=0.007,
):
    print("Attack Image & Predicted Label")

    model.eval()
    correct = 0
    total = 0
    attack_total = 0
    pbar = tqdm(loader, total=len(dataset) // batch_size) if verbose else loader
    for images, labels in pbar:
        for t in range(attack_times):
            images = fgsm_attack(model, loss, images, labels, eps, device).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, pre = torch.max(outputs.data, 1)
            if (pre == labels).sum() == 0:
                break
        else:
            pass
            # print('unbreakable')
        attack_total += t + 1
        correct += (pre == labels).sum()
        total += 1
        if viz:
            imshow(
                torchvision.utils.make_grid(images.cpu().data, normalize=True),
                [idx2label[i] for i in pre],
            )

    print("Accuracy of test text: %f %%" % (100 * float(correct) / total / batch_size))
    print("Avg attack time is: %f " % (attack_total / total))

