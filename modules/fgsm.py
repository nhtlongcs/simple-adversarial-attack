import torch


def fgsm_attack(model, loss, images, labels, eps, device):
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True

    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    attack_images = images + eps * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)

    return attack_images.detach()
