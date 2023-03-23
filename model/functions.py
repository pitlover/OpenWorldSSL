def freeze_layers(model):
    for name, param in model.named_parameters():
        if ('linear' not in name or 'fc' not in name) and 'layer4' not in name:
            param.requires_grad = False

    return model
