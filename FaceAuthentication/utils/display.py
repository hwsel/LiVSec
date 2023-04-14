import matplotlib.pyplot as plt
import torch


def to_numpy(x):
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    # Scale back to range [0, 1]
    x = (x * std) + mean
    x = x.squeeze(0).permute(1, 2, 0)
    return x.numpy()


def display(x_ref, x, cosine_x):

    x_ref = to_numpy(x_ref)
    x = to_numpy(x)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(10, 10)

    axs[0, 0].imshow(x_ref[:, :, :3])
    axs[0, 0].set_title("x_ref - RGB")
    axs[1, 0].imshow(x_ref[:, :, 3], cmap="RdYlBu")
    axs[1, 0].set_title("x_ref - Depth")

    axs[0, 1].imshow(x[:, :, :3])
    axs[0, 1].set_title("x - RGB \n Cosine to x_ref: " + str(cosine_x), color="b")
    axs[1, 1].imshow(x[:, :, 3], cmap="RdYlBu")
    axs[1, 1].set_title("x - Depth")

    for i in range(2):
        for j in range(2):
            axs[i, j].axis("off")

    plt.show()
