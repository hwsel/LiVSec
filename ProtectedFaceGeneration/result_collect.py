from main_module import AttackFaceIDModule
from ProtectedFaceGeneration.modules.data_module import AttackFaceIDDataset, AttackFaceIDDataModule
from modules.faceid_module import FaceIDModule
from torch.utils.data import DataLoader
from utils.display import display
from argparse import ArgumentParser
import torch
import matplotlib.pyplot as plt

import numpy as np
import lpips

def to_numpy(x):
    mean = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1).cuda()
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1).cuda()
    # Scale back to range [0, 1]
    x = (x * std) + mean
    x = x.squeeze(0).permute(1, 2, 0)
    return x.cpu().numpy()

def unnormalize(x):
    mu = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4, 1, 1)
    mu = mu.to(x.device)
    std = std.to(x.device)
    x = (x * std) + mu
    return x


parser = ArgumentParser()

# PROGRAM level args
parser.add_argument("--description", type=str, default="AttackFaceID")
parser.add_argument("--data_path", type=str, default="/data/faceid")

# MODULE specific args
parser = AttackFaceIDModule.add_model_specific_args(parser)

# DATA specific args
parser = AttackFaceIDDataModule.add_data_specific_args(parser)

# TRAINER args
parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--no_workers", type=int, default=0)
parser.add_argument("--max_epochs", type=int, default=200)
parser.add_argument("--threshold", type=float, default=0.9)
args = parser.parse_args([])

dataset = AttackFaceIDDataset(args, split="val", mode="result_collect")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataloader = iter(dataloader)

# Load best checkpoint
faceid = FaceIDModule.load_from_checkpoint("modules/livsec_face_auth.ckpt")
faceid.cuda()
faceid.freeze()

attack_faceid = AttackFaceIDModule.load_from_checkpoint("modules/livsec_protect.ckpt")
attack_faceid.cuda()
attack_faceid.freeze()

cosine = torch.nn.CosineSimilarity()
loss_fn = lpips.LPIPS(net='alex').cuda()
raw_similarity_list = []
adv_similarity_list = []
l2_norm_list = []
lpips_list = []
cnt = 0
for user in range(5):
    i = 0
    while i < 51:
        batch = next(dataloader)
        for j in range(len(batch)):
            batch[j] = batch[j].cuda()

        i = i + 1
        if i == 19:
            continue

        x_ref, x, x_mask = batch

        with torch.no_grad():
            x_adv = attack_faceid(x, x_mask)

            embed_x_ref = faceid(x_ref)
            embed_x = faceid(x)
            embed_x_adv = faceid(x_adv)

            cosine_x = round(cosine(embed_x_ref, embed_x).item(), 3)
            cosine_x_adv = round(cosine(embed_x_ref, embed_x_adv).item(), 3)

            lpips = round(loss_fn.forward(unnormalize(x_adv)[:,:3,:,:], unnormalize(x)[:,:3,:,:]).item(), 4)

        x_numpy = to_numpy(x)
        x_adv_numpy = to_numpy(x_adv)
        noise = abs(x_adv_numpy - x_numpy)
        l2 = np.linalg.norm(noise) / np.linalg.norm(x_numpy)

        raw_similarity_list.append(cosine_x)
        adv_similarity_list.append(cosine_x_adv)
        l2_norm_list.append(l2)
        lpips_list.append(lpips)

        # if (i-1) % 10 == 1:
        #     display(x_ref, x, x_adv, cosine_x, cosine_x_adv, l2, lpips, user, i-1, save=True)


print('avg raw similarity', np.mean(raw_similarity_list))
print('avg adv similarity', np.mean(adv_similarity_list))
print('avg l2 norm', np.mean(l2_norm_list))
print('avg lpips', np.mean(lpips_list))

print('lpips min', np.min(lpips_list))
print('lpips first quartile', np.quantile(lpips_list, 0.25))
print('lpips medium', np.median(lpips_list))
print('lpips 3rd qurtile', np.quantile(lpips_list, 0.75))
print('lpips max', np.max(lpips_list))
print('lpips ave', np.mean(lpips_list))

print('l2 min', np.min(l2_norm_list))
print('l2 first quartile', np.quantile(l2_norm_list, 0.25))
print('l2 medium', np.median(l2_norm_list))
print('l2 3rd qurtile', np.quantile(l2_norm_list, 0.75))
print('l2 max', np.max(l2_norm_list))
print('l2 ave', np.mean(l2_norm_list))

valid_case = 0
success_case = 0
threshold = 0.9
for i in range(len(raw_similarity_list)):
    if raw_similarity_list[i] >= threshold:
        valid_case = valid_case + 1
        if adv_similarity_list[i] < threshold:
            success_case = success_case + 1
print(success_case, valid_case, success_case / valid_case)

plt.hist(raw_similarity_list, range=[0, 1], bins=100)
plt.hist(adv_similarity_list, range=[0, 1], bins=100)
plt.title('Dataset #1. Similarity Score Distribution w/ and w/o $\it{LiVSec}$.')
plt.legend(['Similarity Score w/o $\it{LiVSec}$', 'Similarity Score w/ $\it{LiVSec}$'], fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Similarity Score", fontsize=14)
plt.ylabel("Number of Cases", fontsize=14)
plt.show()
