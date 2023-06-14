'''
The majority of the code is available at https://github.com/torrvision/focal_calibration/blob/main/train_utils.py
Some extra code is added to update the gamma-parameter used by the Adafocal loss function.
'''

import torch
from utils.eval_utils import evaluate_dataset

def train_single_epoch(args,
                        epoch,
                        train_loader,
                        val_loader,
                        num_labels,
                        model,
                        loss_function,
                        optimizer,
                        device):
    model.train()
    num_samples, train_loss = 0, 0
    for batch_idx, batch in enumerate(train_loader):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(data)
        loss = loss_function(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()

        train_loss += loss.item()
        num_samples += len(data)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader)*len(data),
                100.*batch_idx/len(train_loader), loss.item()))

        # This updates the Adafocal's gamma-parameter either after every epoch or after a specified number of batches.
        if args.loss == "adafocal" and args.update_gamma_every == -1 and batch_idx == len(train_loader)-1:
            print("Gamma updated after the end of epoch.")
            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_labels)
            loss_function.update_bin_stats(val_adabin_dict)
        elif args.loss == "adafocal" and args.update_gamma_every > 0 and batch_idx > 0 and batch_idx % args.update_gamma_every == 0:
            print("Gamma updated after batch:", batch_idx)
            (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
            val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_labels)
            loss_function.update_bin_stats(val_adabin_dict)

    train_loss = train_loss/num_samples
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss, loss_function