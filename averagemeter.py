class AverageMeter(object):
    """
    Class for keeping track of averages. In this project, we use it for average loss across an entire data set
    inspired by/adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L363
    """

    def __init__(self, name, format=':f'):
        self.name = name,
        self.format = format
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
        fmtstr = f"{self.name} {self.val}{self.format} ({self.avg}{self.format})"
        return fmtstr
