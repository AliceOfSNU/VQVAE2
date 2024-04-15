import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def plot_tensor(t:torch.Tensor):
    fig, ax = plt.subplots(1, 1)
    if len(t.shape) > 3: t = t[0]
    elif len(t.shape) == 2: t.unsqueeze(0)
    ax.imshow(t.detach().permute(1, 2, 0).cpu().numpy())
    
def save_numpy_images(t, save_path):
    if len(t.shape) > 3:
        datas = [ex for ex in t]
    else:
        datas = [t]
        
    for idx, data in enumerate(datas):
        im = Image.fromarray(np.uint8(data*255))
        path = os.path.join(save_path, f"img{idx}.jpeg")
        im.save(path, "jpeg")
        
def plt_to_numpy(t, plotter):
    # given a numpy array, plot with plt and return the plots as numpy
    datas = [ex for ex in t]
    ret = []
    for img in datas:
        fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        plotter(ax, img)
        plt.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        ret.append(data)
        plt.close(fig)
    return ret


def plot_tensor_grid(t:torch.Tensor, grid:tuple[int, int], 
                     save_to=None, title=None, xlabel=None, ylabel=None):
    t = t.detach().permute(0, 2, 3, 1).cpu().numpy()
    fig, ax = plt.subplots(*grid)
    for i in range(grid[0]):
        for j in range(grid[1]):
            ax[i, j].imshow(t[i*grid[0]+j])
    if save_to is not None:
        fig.savefig(save_to)
        
def load_model(load_path, model, optimizer=None):

    print(f"Loading checkpoint from {load_path}")
    checkpoint  = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict= False)
    checkpoint.pop('model_state_dict')
    
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint.pop('optimizer_state_dict')

    # return the rest in a dict
    return [model, optimizer, checkpoint]
    
def save_model(save_path, epoch, model, optimizer, lr_scheduler=None, stats_dict=None):
    dict =  {'model_state_dict'        : model.state_dict(),
            'optimizer_state_dict'     : optimizer.state_dict(),
            'epoch'                    : epoch
            }
    if lr_scheduler:
        dict['scheduler_state_dict'] = lr_scheduler.state_dict()
    if stats_dict:
        dict.update(stats_dict)
        
    filename = "epoch_{}.pth".format(epoch)
    torch.save(dict, os.path.join(save_path, filename))
    