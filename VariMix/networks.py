import torch.nn as nn

from torchvision import models


def define_pretrained_model(model_name, num_classes):
    """
    The following classification models are available, 
    with or without pre-trained weights:
    """
    model = None

    #--------------------------------------VGG (2014)--------------------------------------#
    if model_name == 'vgg13':     
        model = models.vgg13(weights = models.VGG13_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features            
        model.classifier[6] = nn.Linear(num_fc, num_classes) 
    if model_name == 'vgg16':     
        model = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features            
        model.classifier[6] = nn.Linear(num_fc, num_classes) 

    #-----------------------------------GoogleNet (2014)-----------------------------------#
    if model_name == 'googlenet':
        model = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)

    #-------------------------------------ResNet (2015)------------------------------------#
    if model_name == 'resnet18':
        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes) 

    if model_name == 'resnet34':
        model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)

    if model_name == 'resnet50':
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes) 
        
    #------------------------------------DenseNet (2016)-----------------------------------#
    if model_name == 'densenet121':
        model = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
        model.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
        
    #------------------------------------MobileNet (2019)----------------------------------#
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes)

    #-------------------------------------ConvNeXt (2022)----------------------------------#
    if model_name == 'convnext_tiny':  # tiny, small, base, large
        model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_fc = model.classifier[2].in_features 
        model.classifier[2] = nn.Linear(num_fc, out_features = num_classes)

    if model_name == 'vit_b_16':
        model = models.vit_b_16(weights = models.ViT_B_16_Weights.DEFAULT)   
        num_fc = model.heads.head.in_features 
        model.heads.head = nn.Linear(num_fc, out_features = num_classes)

    if model_name == 'swin_v2_t':
        model = models.swin_v2_t(weights = models.Swin_V2_T_Weights.DEFAULT)   
        num_fc = model.head.in_features 
        model.head = nn.Linear(num_fc, out_features = num_classes)
    return model
