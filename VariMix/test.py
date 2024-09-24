import torch
import argparse
import os
import torch.nn as nn
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from datasets import TestLoader
from networks import define_pretrained_model
from feature_show import define_model_trunc, plot_2d_features, plot_3d_features, plot_confusion_matrix


def test_model(model, test_loader, args):
    test_correct = 0    # keep track of testing correct numbers
    y_prob = []         # getting AUC
    encoding_array = [] # tsne
    model_trunc = define_model_trunc(args.model_name, model) # tsne

    classes = list(test_loader.classID) # accuracy for each category & confusion matrix
    pred_cm = torch.tensor([], dtype=float, device='cuda')
    pred_cm = pred_cm.cuda(args.gpu_ids[0])  # for confusion matrix

    if model_trunc == None:
        print('define_model_{}_trunc failed !'.format(args.model_name))
        exit(0)

    for i, (data, target) in enumerate (test_loader.data_loader):
        data, target = data.cuda(args.gpu_ids[0]), target.cuda(args.gpu_ids[0])  # move to GPU or cpu
        output = model(data)  # forward pass: compute predicted outputs by passing inputs to the model

        predict_y = torch.max(output, dim=1)[1]              # output predicted class (i.e., idx)
        test_correct += (predict_y == target).sum().item()   # update validation correct numbers

        prob = torch.softmax(output, dim=1)    # output probabilities for plotting roc curve
        y_prob_ = np.squeeze(prob.data.cpu().float().numpy())
        y_prob.append(y_prob_)

        feature = model_trunc(data)['semantic_feature'].squeeze().detach().cpu().numpy()
        encoding_array.append(feature)

        pred_cm = torch.cat((pred_cm, predict_y), dim=0)

    ave_test_acc = test_correct/len(test_loader.data_loader.sampler)    # calculate average accuracy
    print('Testing Accuracy: {:.4f} ({}/{})'.format(ave_test_acc, test_correct, len(test_loader.data_loader.sampler)))
    
    print('\nThe Confusion Matrix is plotted and saved:')
    cMatrix = confusion_matrix(torch.tensor(test_loader.labels), pred_cm.cpu())
    print(cMatrix)
    
    matrix_path ='features/{}/{}'.format(args.dataset_name, args.save_dirname)
    if not os.path.exists(matrix_path): 
        os.makedirs(matrix_path)
    plot_confusion_matrix(classes, cMatrix, matrix_path)

    if args.test_auc:
        roc_auc_ovr = []
        avg_roc_auc = 0.0
        y_prob = np.array(y_prob)
        # print(y_prob.shape) # (num_testImages, num_classes)
        for v in test_loader.classID.values(): # visit each class idx
            y_true = [1 if y == v else 0 for y in test_loader.labels] # construct list for binary labels
            auc_score = roc_auc_score(y_true, y_prob[:, v])
            avg_roc_auc += auc_score
            roc_auc_ovr.append(auc_score)
        print(f"Testing AUC OvR: {avg_roc_auc/len(test_loader.classID):.4f}")

    encoding_array = np.array(encoding_array)
    class_to_idx = test_loader.classID
    feature_path = 'features/{}/{}'.format(args.dataset_name, args.save_dirname)
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    testset_targets = np.array(test_loader.labels) # The class_index value for each image in the dataset
    plot_2d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    plot_3d_features(encoding_array, class_to_idx, feature_path, testset_targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='/home/xiangyu/data/breast-ultrasound-dataset/test_dir', type=str, help='test data directory')
    parser.add_argument('--json_path', default = '/home/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json',  type=str, help='classid and classname')
    parser.add_argument('--save_dirname', default='swin_v2_t_7', type=str, help='name to save the tsne figure')
    parser.add_argument('--dataset_name', default='breast-ultrasound-dataset', type=str, help='dataset name')
    parser.add_argument('--test_auc', action='store_true', help='if true, test OvR AUC score')
    parser.add_argument('--model_name', default='swin_v2_t', type=str,
        choices=['alexnet', 'vgg16', 'vgg19', 'googlenet', 'resnet18', 'resnet34', 'resnet50', 'densenet121',
        'mobilenet_v3_small', 'efficientnet_b5', 'convnext_tiny','vit_b_16', 'swin_v2_t'], help='Choose the model you want train')
    parser.add_argument('--use_pretrain', action='store_true', help='if specified, use the pretrained models of torchvision')
    parser.add_argument('--image_size', type=int, default=256, help='size of the data')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--test_batchSize', default=1, type=int, help='testing batch size, len(dataset) mod batchsize = 0')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

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
    
    checkpoint_path = 'checkpoints/{}/{}/{}_checkpoint.pth'.format(args.dataset_name, args.save_dirname, args.model_name)
    print(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print("checkpont file does not exist!")
        exit(0)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cuda:0'))
    model.eval()

    test_loader = TestLoader(args.image_size, args.test_dir, args.test_batchSize, args.n_cpu, args.json_path)
    test_model(model, test_loader, args)