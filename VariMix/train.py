import torch
import torchvision
import argparse
import os
import json
import numpy as np
import torch.nn as nn

from utils import variety_mix, loss_visualize, acc_visualize, get_scheduler, update_learning_rate, translate_and_fusion
from datasets import TrainLoader, ValidLoader
from networks import define_pretrained_model
from model import build_model
from checkpoint import CheckpointIO
from os.path import join as ospj


def train_model(model, train_loader, val_loader, nets, args):
    val_loss_min = np.Inf  # track change in minimum validation loss
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []
    
    nets.style_encoder.cuda(args.gpu_ids[0])
    nets.generator.cuda(args.gpu_ids[0])
    nets.style_encoder.eval()
    nets.generator.eval()

    criterion = nn.CrossEntropyLoss()  # define the cost function
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(optimizer, args)

    for epoch in range(args.epoch_count, args.n_epochs + args.n_epochs_decay + 1):
        print('\n#---------------------------------------------running epoch: [{}/{}]---------------------------------------------#'
        .format(epoch, args.n_epochs + args.n_epochs_decay))
        
        #----------------------------------------------- train the model-------------------------------------------------#
        model.train()
        train_loss = 0.0    # keep track of training loss
        train_correct = 0   # keep track of training correct numbers
        for i, (data, target) in enumerate(train_loader.data_loader):
            data, target = data.cuda(args.gpu_ids[0]), target.cuda(args.gpu_ids[0])  # move tensors to GPU or cpu
            data_src, data_ref, target_src, target_ref, lam = variety_mix(args, data, target, 1.0, nets)
            # translate_and_fusion(nets, data, target, 'MixRS.png')
            # exit(0)
            
            optimizer.zero_grad()             # clear the gradients of all optimized variables
            output_src = model(data_src)      # forward pass: compute predicted outputs by passing inputs to the model
            output_ref = model(data_ref)      # forward pass: compute predicted outputs by passing inputs to the model
            loss_src = torch.mean((1-lam) * criterion(output_src, target_src) + lam * criterion(output_src, target_ref))  # calculate the mixup batch loss
            loss_ref = torch.mean(lam * criterion(output_ref, target_src) + (1-lam) * criterion(output_ref, target_ref))  # calculate the mixup batch loss
            loss = (loss_src + loss_ref) / 2
            loss.backward()   # backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # perform a single optimization step (parameter update)

            # loss.item()is the “averaged across all training examples of the current batch”
            # loss.item()*data.size(0) is the “total loss of the current batch (not averaged)”
            train_loss += loss.item()*data.size(0)  # item(), get a Python number from a tensor containing a single value.

            predict_y_src = torch.max(output_src, dim=1)[1] 
            predict_y_ref = torch.max(output_ref, dim=1)[1] 
            train_correct += ((predict_y_src == target).sum().item() + (predict_y_ref == target).sum().item()) / 2 # update training correct numbers

        update_learning_rate(optimizer, scheduler)  # update learning rates
        
        #---------------------------------------------- validate the model ----------------------------------------------#
        model.eval()
        val_loss = 0.0    # keep track of validation loss
        val_correct = 0   # keep track of validation correct numbers
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader.data_loader):
                data, target = data.cuda(args.gpu_ids[0]), target.cuda(args.gpu_ids[0])  # move tensors to GPU or cpu
                
                output = model(data)              # forward pass: compute predicted outputs by passing inputs to the model
                loss = criterion(output, target)  # calculate the batch loss
                val_loss += loss.item()*data.size(0)    # update validation loss

                predict_y = torch.max(output, dim=1)[1]
                val_correct += (predict_y == target).sum().item()   # update validation correct numbers
        
        #----------------------------------------- each epoch loss and accuracy -----------------------------------------#
        ave_train_loss = train_loss / len(train_loader.data_loader.sampler)  # calculate average loss
        ave_val_loss = val_loss / len(val_loader.data_loader.sampler)
        ave_train_acc = train_correct/ len(train_loader.data_loader.sampler) # calculate average accuracy
        ave_val_acc = val_correct / len(val_loader.data_loader.sampler)

        print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(ave_train_loss, ave_val_loss))
        print('Training Accuracy: {:.4f} \tValidation Accuracy: {:.4f}'.format(ave_train_acc, ave_val_acc))

        #----------------------------------------- show epoch loss and accuracy -----------------------------------------#
        train_loss_list.append(ave_train_loss)
        val_loss_list.append(ave_val_loss)
        train_acc_list.append(ave_train_acc)
        val_acc_list.append(ave_val_acc)

        result_path = 'results/{}/{}'.format(args.dataset_name, args.save_dirname)
        if not os.path.exists(result_path): 
            os.makedirs(result_path)
        checkpoint_path = 'checkpoints/{}/{}'.format(args.dataset_name, args.save_dirname)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        loss_visualize(epoch, train_loss_list, val_loss_list, result_path)
        acc_visualize(epoch, train_acc_list, val_acc_list, result_path)

        if ave_val_loss < val_loss_min: # save model if validation loss has decreased
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, ave_val_loss))
            torch.save(model.state_dict(), checkpoint_path + '/{}_checkpoint.pth'.format(args.model_name))
            val_loss_min = ave_val_loss


def load_checkpoint(step):
    for ckptio in ckptios:
        ckptio.load(step)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='/home/gem/xiangyu/data/breast-ultrasound-dataset/train_dir', type=str, help='train data directory')
    parser.add_argument('--valid_dir', default='/home/gem/xiangyu/data/breast-ultrasound-dataset/valid_dir', type=str, help='validation data directory')
    parser.add_argument('--json_path', default = '/home/gem/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json',  type=str, help='classid and classname')
    parser.add_argument('--save_dirname', default='convnext_tiny1', type=str, help='name to save the checkpoint and result figure')
    parser.add_argument('--dataset_name', default='breast-ultrasound-dataset', type=str, help='dataset name')
    parser.add_argument('--model_name', default='convnext_tiny', type=str, 
        choices=['alexnet', 'vgg16','resnet18', 'resnet34', 'resnet50', 'convnext_tiny', 'vit_b_16', 'swin_v2_t'], help='Choose the model you want train')
    parser.add_argument('--image_size', type=int, default=256, help='size of the data')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--train_batchSize', type=int, default=64, help='size of the batches')
    parser.add_argument('--val_batchSize', type=int, default=64, help='size of the batches')
    parser.add_argument('--learning_rate', default=2e-4, type=float, help='initial learning rate for adam')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
    parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | cosine]')
    parser.add_argument('--gpu_ids', type=str, default='0, 1')
    
    # use stargan-v2 networks
    parser.add_argument('--stargan_size', type=int, default=256, help='size used in stargan-v2 (squared assumed)')
    parser.add_argument('--num_domains', type=int, default=3, help='Number of domains')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
    parser.add_argument('--style_dim', type=int, default=64, help='Style code dimension')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints', help='Directory for saving network checkpoints')
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')

    args = parser.parse_args()

    if not os.path.exists(args.train_dir):
        print("train dataset directory not exists")
        exit(0)

    if not os.path.exists(args.valid_dir):
        print("valid dataset directory not exists")
        exit(0)

    if torch.cuda.is_available():
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)   
    else:
        print('CUDA is not available!')
        exit(0)
    
    model = define_pretrained_model(args.model_name, args.num_classes)
    if model == None:
        print('Model {} is not constructed success!'.format(args.model_name))
        exit(0)
    print('Model {} is constructed success!'.format(args.model_name))

    model.cuda(args.gpu_ids[0])
    model = nn.DataParallel(model, args.gpu_ids)

    with open(args.json_path, 'r') as f:
        cat2name_mapping = json.load(f)
        if args.num_classes != len(cat2name_mapping):
            print("The argument num_classes is not equal to the real classes!")
            exit(0)

    if 'cifar' in args.dataset_name and args.image_size != 32:
        print("cifar image size is wrong!")
        exit(0)

    train_loader = TrainLoader(args.image_size, args.train_dir, args.train_batchSize, args.n_cpu, args.json_path)
    valid_loader = ValidLoader(args.image_size, args.valid_dir, args.val_batchSize, args.n_cpu, args.json_path)
    print('Train and Validation dataloaders have been finished!')

    nets, nets_ema = build_model(args)
    ckptios = [CheckpointIO(ospj(args.checkpoint_dir, args.dataset_name, '{:06d}_nets_ema.ckpt'), data_parallel=True, **nets_ema)]
    load_checkpoint(args.resume_iter)
    train_model(model, train_loader, valid_loader, nets_ema, args)