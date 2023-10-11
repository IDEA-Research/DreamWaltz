from torch.optim import lr_scheduler
from .lr_cosine import CosineLRScheduler
from .lr_warmup import WarmupMultiStepLR



def make_scheduler(optimizer, lr_policy, gamma=0.1, iterations=-1, step_size=None, milestones=None, warmup_iter=None, lr_lambda=None):
    if lr_policy == 'constant':
        scheduler = None
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, last_epoch=iterations)
    elif lr_policy in ['multi_step', 'multistep']:
        if milestones is None:
            milestones = [step_size, step_size + step_size//2, step_size + step_size//2 + step_size//4]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=iterations)
    elif lr_policy == 'warmup':
        if milestones is None:
            milestones = [step_size, step_size + step_size//2, step_size + step_size//2 + step_size//4]
        scheduler = WarmupMultiStepLR(optimizer,
                                      milestones=milestones,
                                      warmup_iter=warmup_iter,
                                      gamma=gamma,
                                      last_epoch=iterations)
    elif lr_policy == 'lambda':
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise NotImplementedError('lr policy [%s] is not implemented', lr_policy)
    return scheduler
