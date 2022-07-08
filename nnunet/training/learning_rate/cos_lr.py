import math

def cos_lr(initial_lr,epoch,min_lr,Tmax):
    return min_lr+(initial_lr-min_lr)*(1+math.cos(epoch/Tmax*math.pi))/2


