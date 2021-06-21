if __name__ == "__main__":
    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, pre = torch.max(outputs.data, 1)

        total += imgnet_dataloader.batch_size
        correct += (pre == labels).sum()
