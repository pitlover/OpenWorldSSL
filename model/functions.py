def freeze_layers(model, model_name):
    if "resnet18" in model_name:
        fc = "linear"
    elif "resnet50" in model_name:
        fc = "fc"
    else:
        raise ValueError(f"Not supported Backbone {model_name}.")

    for name, param in model.named_parameters():
        if fc not in name and 'layer4' not in name:
            param.requires_grad = False
    return model


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
