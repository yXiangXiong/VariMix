import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def define_model_trunc(model_name, model):
    model_trunc = None
    if model_name == 'alexnet':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'classifier.4': 'semantic_feature'})
    if model_name == 'googlenet':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'avgpool': 'semantic_feature'})
    if model_name == 'resnet18' or model_name == 'resnet34' or model_name == 'resnet50':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'avgpool': 'semantic_feature'})
    if model_name == 'vgg16' or model_name == 'vgg19':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'classifier.3': 'semantic_feature'})
    if model_name == 'mobilenet_v2':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'classifier.0': 'semantic_feature'})
    if model_name == 'densenet121' or model_name == 'densenet161':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'features.avgpool': 'semantic_feature'})
    if model_name == 'efficientnet_b5':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'avgpool': 'semantic_feature'})
    if model_name == 'convnext_tiny':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'avgpool': 'semantic_feature'})
    if model_name == 'vit_b_16':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'heads.head': 'semantic_feature'})
    if model_name == 'swin_v2_t':
        model_trunc = create_feature_extractor(model.module, return_nodes = {'avgpool': 'semantic_feature'})
    return model_trunc


def plot_2d_features(encoding_array, class_to_idx, feature_path, targets):
    marker_list = ['>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', '.', ',', 'o', 'v', '^', '<',  'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    n_class = len(class_to_idx)          
    palette = sns.hls_palette(n_class)
    sns.palplot(palette)

    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)

    tsne = TSNE(n_components=2, learning_rate='auto', n_iter=20000, init='pca')
    X_tsne_2d = tsne.fit_transform(encoding_array)
    
    plt.figure(figsize=(14, 14))
    
    for key, value in class_to_idx.items(): # {'cat': 0, 'dog': 1}
        color = palette[value]
        marker = marker_list[value % len(marker_list)]
    
        indices = np.where(targets==value)
        plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=key, s=150)
    
    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    plt.savefig(feature_path + '/2d_t-sne.png', dpi=200)
    plt.show()


def plot_3d_features(encoding_array, class_to_idx, feature_path, targets):
    marker_list = ['>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', '.', ',', 'o', 'v', '^', '<',  'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    n_class = len(class_to_idx)          
    palette = sns.hls_palette(n_class)
    sns.palplot(palette)

    random.seed(1234)
    random.shuffle(marker_list)
    random.shuffle(palette)

    tsne = TSNE(n_components=3, learning_rate='auto', n_iter=10000, init='pca')
    X_tsne_3d = tsne.fit_transform(encoding_array)

    fig = plt.figure(figsize=(14, 14))
    ax = Axes3D(fig)

    for key, value in class_to_idx.items(): # {'cat': 0, 'dog': 1}
        color = palette[value]
        marker = marker_list[value % len(marker_list)]
    
        indices = np.where(targets==value)
        ax.scatter(X_tsne_3d[indices, 0], X_tsne_3d[indices, 1], X_tsne_3d[indices, 2], color=color, marker=marker, label=key, s=150)
    
    plt.legend(fontsize=16, markerscale=1, bbox_to_anchor=(1, 1))
    plt.savefig(feature_path + '/3d_t-sne.png', dpi=100)
    plt.show()


def plot_confusion_matrix(class_names, confusion_matrix, result_path):
    cfmt =pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(12, 10), dpi=250)
    plt.title('Confusion Matrix')
    ax= sns.heatmap(cfmt, annot=True, cmap='BuGn', fmt="d")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(result_path + '/confusion_matrix.png')