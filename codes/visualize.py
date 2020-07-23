import matplotlib.pyplot as plt
from skimage import color


__all__ = ['visualize_attribution_maps']


def visualize_attribution_maps(data, fpath):
    fig, axes = plt.subplots(ncols=3)
    fig.set_size_inches(15, 5)

    image = data['image']
    bg = color.rgb2gray(image)

    ax = axes[0]
    ax.imshow(data['image'])
    ax.set_axis_off()

    ax = axes[1]
    igmap = data['IGmap']
    vmax = abs(igmap).max()
    vmin = 0
    ax.imshow(bg)
    ax.imshow(igmap, cmap='gray', interpolation='nearest', vmax=vmax, vmin=vmin, alpha=0.8)
    ax.set_axis_off()

    ax = axes[2]
    pmimap = data['PMImaps'][0]
    vmax = abs(pmimap).max()
    vmin = -vmax
    ax.imshow(bg)
    ax.imshow(pmimap, cmap='RdBu_r', interpolation='nearest', vmax=vmax, vmin=vmin, alpha=0.8)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(fpath)
