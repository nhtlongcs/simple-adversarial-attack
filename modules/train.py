import time
import copy

import torch

from tqdm.notebook import tqdm

from .fgsm import fgsm_attack
from .utils import device

def adversarial_examples(model, criterion, inputs, labels, eps, device):
    model.eval()
    adv_inputs = fgsm_attack(model, criterion, inputs, labels, eps, device)
    model.train()

    return adv_inputs

def adversarial_train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, eps=0.007, alpha=0.5):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        _loss = loss1 + 0.4*loss2

                        # generate adversarial examples
                        adv_inputs = adversarial_examples(model, criterion, inputs, labels, eps, device)

                        adv_outputs, adv_aux_outputs = model(adv_inputs)
                        adv_loss1 = criterion(adv_outputs, labels)
                        adv_loss2 = criterion(adv_aux_outputs, labels)
                        adv_loss = adv_loss1 + 0.4*adv_loss2

                        loss = alpha * _loss + (1 - alpha) * adv_loss
                    else:
                        outputs = model(inputs)
                        _loss = criterion(outputs, labels)

                        # generate adversarial examples
                        adv_inputs = adversarial_examples(model, criterion, inputs, labels, eps, device)
                        
                        adv_outputs = model(adv_inputs)
                        adv_loss = criterion(adv_outputs, labels)

                        loss = alpha * _loss + (1 - alpha) * adv_loss


                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history