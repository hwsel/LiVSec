import matplotlib.pyplot as plt
import torch


def to_numpy(x):
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1).cuda()
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1).cuda()
    # Scale back to range [0, 1]
    x = (x * std) + mean
    x = x.squeeze(0).permute(1, 2, 0)
    return x.cpu().numpy()


def display(x_ref, x, x_adv, cosine_x, cosine_x_adv, l2, lpips, usr, idx, save=False):

    x_ref = to_numpy(x_ref)
    x = to_numpy(x)
    x_adv = to_numpy(x_adv)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)

    axs[0, 0].imshow(x_ref[:, :, :3])
    axs[0, 0].set_title("X_ref - RGB")
    axs[1, 0].imshow(x_ref[:, :, 3], cmap="RdYlBu")
    axs[1, 0].set_title("X_ref - Depth")

    axs[0, 1].imshow(x[:, :, :3])
    axs[0, 1].set_title("X - RGB \n Cosine to X_ref: " + str(cosine_x), color="b")
    axs[1, 1].imshow(x[:, :, 3], cmap="RdYlBu")
    axs[1, 1].set_title("X - Depth")

    axs[0, 2].imshow(x_adv[:, :, :3])
    axs[0, 2].set_title(
        "X_adv - RGB \n Cosine to X_ref: " + str(cosine_x_adv) +
        "\nL2: " + str(l2)[:6] + '\nLPIPS: ' + str(lpips), color="b"
    )
    axs[1, 2].imshow(x_adv[:, :, 3], cmap="RdYlBu")
    axs[1, 2].set_title("X_adv - Depth")

    for i in range(2):
        for j in range(3):
            axs[i, j].axis("off")

    if save:
        plt.savefig('lpips_results/dataset1_'+str(usr)+'_'+str(idx)+'.png')
        plt.close()
    else:
        plt.show()


def display2(x_ref, x, x_adv, cosine_x, cosine_x_adv, l2, l1):
    x_ref = to_numpy(x_ref)
    x = to_numpy(x)
    x_adv = to_numpy(x_adv)

    fig, axs = plt.subplots(2, 3)
    fig.set_size_inches(15, 10)

    axs[0, 0].imshow(x_ref[:, :, :3])
    axs[0, 0].set_title("X_ref - RGB")
    axs[1, 0].imshow(x_ref[:, :, 3], cmap="gray")
    axs[1, 0].set_title("X_ref - Depth")

    axs[0, 1].imshow(x[:, :, :3])
    axs[0, 1].set_title("X - RGB \n Cosine to X_ref: " + str(cosine_x), color="b")
    axs[1, 1].imshow(x[:, :, 3], cmap="gray")
    axs[1, 1].set_title("X - Depth")

    axs[0, 2].imshow(x_adv[:, :, :3])
    axs[0, 2].set_title(
        "X_adv - RGB \n Cosine to X_ref: " + str(cosine_x_adv) + "\nL2: " + str(l2)[:5]
        + "\n % of d change: " + str(l1)[:5], color="b"
    )
    axs[1, 2].imshow(x_adv[:, :, 3], cmap="gray")
    axs[1, 2].set_title("X_adv - Depth")

    for i in range(2):
        for j in range(3):
            axs[i, j].axis("off")

    plt.show()
