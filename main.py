from torch import optim
import torch
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import random
import json
import sys
import os
import argparse
import collections

import dataloaders.cifar10 as cifar10 # CIFAR-10 is an image dataset publicly available at https://www.cs.toronto.edu/~kriz/cifar.html
from models.resnet import resnet50 # ResNet-50 is a deep residual network whose implementation is publicly available at https://github.com/KaimingHe/deep-residual-networks
from losses.adafocal import AdaFocal # Adafocal is our proposed loss function from the paper https://proceedings.neurips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html
from utils.train_utils import train_single_epoch
from utils.eval_utils import evaluate_dataset

dataset_loader = {
    'cifar10': cifar10, 
}
dataset_num_classes = { 
    'cifar10': 10, 
}
models = {
    'resnet50': resnet50, 
}

# Definition of various arguments to be passed to the main.py training script.
def parseArgs():
    dataset = 'cifar10'
    dataset_root = './data'
    train_batch_size = 128
    test_batch_size = 128

    learning_rate = 0.1
    momentum = 0.9
    optimiser = "sgd"
    weight_decay = 5e-4
    num_epochs = 350
    first_milestone = 150 # Milestone for change in lr
    second_milestone = 250 # Milestone for change in lr
    loss = "cross_entropy"
    log_interval = 10
    save_interval = 50

    model = "resnet50"
    model_checkpoint = None
    save_path = './exp'
    load_path = './exp'

    num_bins = 15
    adafocal_lambda = 1.0
    adafocal_gamma_initial = 1.0
    adafocal_gamma_max = 20.0
    adafocal_gamma_min = -2.0
    adafocal_switch_pt = 0.2
    update_gamma_every = -1

    parser = argparse.ArgumentParser(description="Adafocal training algorithm.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, dest="seed", help="torch.manual_seed()")
    parser.add_argument("--dataset", type=str, default=dataset, dest="dataset", help='dataset to train on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root, dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--data-aug", action="store_true", dest="data_aug")
    parser.set_defaults(data_aug=True)

    parser.add_argument("-g", action="store_true", dest="gpu", help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-b", type=int, default=train_batch_size, dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size, dest="test_batch_size", help="Test Batch size")
    parser.add_argument("-e", type=int, default=num_epochs, dest="num_epochs", help='Number of training epochs')
    parser.add_argument("--lr", type=float, default=learning_rate, dest="learning_rate", help='Learning rate')
    parser.add_argument("--mom", type=float, default=momentum, dest="momentum", help='Momentum')
    parser.add_argument("--nesterov", action="store_true", dest="nesterov", help="Whether to use nesterov momentum in SGD")
    parser.set_defaults(nesterov=False)
    parser.add_argument("--decay", type=float, default=weight_decay, dest="weight_decay", help="Weight Decay")
    parser.add_argument("--opt", type=str, default=optimiser, dest="optimiser", help='Choice of optimisation algorithm')
    parser.add_argument("--first-milestone", type=int, default=first_milestone, dest="first_milestone", help="First milestone to change lr")
    parser.add_argument("--second-milestone", type=int, default=second_milestone, dest="second_milestone", help="Second milestone to change lr")
    parser.add_argument("--loss", type=str, default=loss, dest="loss", help="Loss function to be used for training")
    parser.add_argument("--log-interval", type=int, default=log_interval, dest="log_interval", help="Log interval for display on terminal.")
    parser.add_argument("--save-interval", type=int, default=save_interval, dest="save_interval", help="Save interval for model checkpoints.")

    parser.add_argument("--model", type=str, default=model, dest="model", help='Model to train')
    parser.add_argument("--save-path", type=str, default=save_path, dest="save_path", help='Path to export the model.')
    parser.add_argument("--load", action="store_true", dest="load", help="Load from pretrained model")
    parser.set_defaults(load=False)
    parser.add_argument("--load-path", type=str, default=load_path, dest="load_path", help='Path to load the model from.')
    parser.add_argument("--model-checkpoint", type=str, default=model_checkpoint, dest="model_checkpoint", help="file name of the pre-trained model.")

    # Adafocal
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins", help="Number of calibration bins")
    parser.add_argument("--adafocal-lambda", type=float, default=adafocal_lambda, dest="adafocal_lambda", help="lambda for adafocal.")
    parser.add_argument("--adafocal-gamma-initial", type=float, default=adafocal_gamma_initial, dest="adafocal_gamma_initial", help="Initial gamma for each bin.")
    parser.add_argument("--adafocal-gamma-max", type=float, default=adafocal_gamma_max, dest="adafocal_gamma_max", help="Maximum cutoff value for gamma.")
    parser.add_argument("--adafocal-gamma-min", type=float, default=adafocal_gamma_min, dest="adafocal_gamma_min", help="Minimum cutoff value for gamma.")
    parser.add_argument("--adafocal-switch-pt", type=float, default=adafocal_switch_pt, dest="adafocal_switch_pt", help="Gamma at which to switch to inverse-focal loss.")
    parser.add_argument("--update-gamma-every", type=int, default=update_gamma_every, dest="update_gamma_every", help="Update gamma every nth batch. If -1, update after epoch end.")

    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(args.seed)
    
    # Get the arguments
    args = parseArgs()
    
    # This selects the GPU to be used for training.
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    cuda = False
    if torch.cuda.is_available() and args.gpu: 
        cuda = True
    device = torch.device("cuda" if cuda else "cpu")
    print("GPU detected: " + str(cuda))

    # This creates the directory to save the experiment.
    if not os.path.isdir(args.save_path): 
        os.makedirs(args.save_path, exist_ok=True)

    # This initializes the model to be trained.
    num_classes = dataset_num_classes[args.dataset]
    model = models[args.model](num_classes=num_classes)
    if args.gpu:
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    
    # In order to resume training from a saved model, this loads the model checkpoint.
    start_epoch = 0
    if args.load:
        model.load_state_dict(torch.load(args.load_path + '/' + args.model_checkpoint))
        start_epoch = int(args.model_checkpoint[args.model_checkpoint_name.rfind('_')+1:args.model_checkpoint.rfind('.model')])

    # This creates the training, validation and test dataloaders
    # whose implementation is available publicly at https://github.com/torrvision/focal_calibration.
    train_loader, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        random_seed=1,
        pin_memory=args.gpu)
    test_loader = dataset_loader[args.dataset].get_test_loader(
        batch_size=args.test_batch_size,
        pin_memory=args.gpu)

    # This initializes the optimiser and the learning rate scheduler which are available from the Pytorch library.
    if args.optimiser == "sgd":
        opt_params = model.parameters()
        optimizer = optim.SGD(opt_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1)
    
    # If training is being resumed from a saved checkpoint, set to the correct learning rate.
    for epoch in range(0, start_epoch):
        lr_scheduler.step()

    # This initializes the loss function object. Cross entropy loss is available from the Pytorch library.
    # AdaFocal is our proposed loss function implemented in losses/adafocal.py
    if args.loss == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss(reduction="sum")
    elif args.loss == "adafocal":
        loss_function = AdaFocal(args, device=device)

    # This is the model traininig loop that runs for a specified number of epochs.
    best_val_acc = 0
    for epoch in range(start_epoch, args.num_epochs):
        # For every epoch, this calls the "train_single_epoch" function implemented in utils/train_utils.py and this code is
        # available at https://github.com/torrvision/focal_calibration
        train_loss, loss_function = train_single_epoch( args=args,
                                                    epoch=epoch,
                                                    train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    num_labels=num_classes,
                                                    model=model,
                                                    loss_function=loss_function,
                                                    optimizer=optimizer,
                                                    device=device)
        lr_scheduler.step()

        # This evaluates the current model on the validation set to collect various performance statistics.
        # This calls the "evaluate_dataset" function implemented in utils/eval_utils.py
        (val_loss, val_confusion_matrix, val_acc, val_ece, val_bin_dict,
        val_adaece, val_adabin_dict, val_mce, val_classwise_ece) = evaluate_dataset(model, val_loader, device, num_bins=args.num_bins, num_labels=num_classes)
        
        # This evaluates the current model on the test set to collect various performance statistics.
        # This calls the "evaluate_dataset" function implemented in utils/eval_utils.py
        (test_loss, test_confusion_matrix, test_acc, test_ece, test_bin_dict, 
        test_adaece, test_adabin_dict, test_mce, test_classwise_ece) = evaluate_dataset(model, test_loader, device, num_bins=args.num_bins, num_labels=num_classes)

        # This saves the above-computed performance statistics and saves them to a text file.
        output_train_file = os.path.join(args.save_path, "train_log.txt")
        with open(output_train_file, "a") as writer:
            writer.write("%d\t" % (epoch))
            writer.write("%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t" % (train_loss, val_loss, test_loss, 1 - val_acc, val_ece, val_mce, val_classwise_ece, val_adaece))
            writer.write("%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (1 - test_acc, test_ece, test_mce, test_classwise_ece, test_adaece))
            writer.write("\n")

        # This saves the current model as a checkpoint in the experiment directory.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print('New best error: %.4f' % (1 - best_val_acc))
            save_name = args.save_path + '/' + args.model + '_best.model'
            torch.save(model.state_dict(), save_name)
        if (epoch + 1) % args.save_interval == 0:
            save_name = args.save_path + '/' + args.model +  '_' + str(epoch + 1) + '.model'
            torch.save(model.state_dict(), save_name)
