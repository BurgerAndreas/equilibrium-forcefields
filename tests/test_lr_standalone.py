from matplotlib import pyplot as plt

from timm import create_model 
from timm.optim import create_optimizer
from types import SimpleNamespace
from timm.scheduler.cosine_lr import CosineLRScheduler

num_epochs=500

def get_lr_per_epoch(scheduler, num_epoch):
    lr_per_epoch = []
    for epoch in range(num_epoch):
        lr_per_epoch.append(scheduler.get_epoch_values(epoch))
    return lr_per_epoch

model = create_model('resnet34')

args = SimpleNamespace()
args.weight_decay = 0
args.lr = 1e-4
args.opt = 'adam' 
args.momentum = 0.9

optimizer = create_optimizer(args, model)

num_epoch = 10
scheduler = CosineLRScheduler(optimizer, t_initial=num_epoch, t_mul=2, decay_rate=1., lr_min=1e-5, cycle_limit=0)
lr_per_epoch = get_lr_per_epoch(scheduler, num_epochs)

plt.plot([i for i in range(num_epochs)], lr_per_epoch)
plt.show()