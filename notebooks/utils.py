import matplotlib.pyplot as plt
import numpy as np

import torchvision.utils as vutils


def display_grid(images, 
                 nrow=8,
                 figsize=(12,12)):
    fig = plt.figure(figsize=figsize)
    plt.imshow(
        np.transpose(
            vutils.make_grid(images, nrow=nrow, padding=2, normalize=True).cpu(),(1,2,0)
        )
    )
    plt.axis('off')