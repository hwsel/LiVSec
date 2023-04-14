import os
import subprocess as sp
import time
from argparse import ArgumentParser

import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch.nn
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
import lpips

from modules.faceid_module import FaceIDModule
from modules.attack_module import AttackFaceIDModule
from modules.data_module import AttackFaceIDDataModuleLiVSec, AttackFaceIDDatasetLiVSec

from utils import get_face_rgbd
from utils import display3
from utils import display
from utils import to_numpy
from utils import unnormalize
from utils import SSIM


STREAMING_MODE, RESULT_COLLECT_MODE, VERBOSE_MODE = range(3)
MODE = RESULT_COLLECT_MODE
REUSE_FREQ = 1

FFMPEG_BIN = '/usr/bin/ffmpeg'

DEFAULT_VIDEO = './John_3dts.mp4'

input_command = [FFMPEG_BIN,
                 '-re',
                 '-i', DEFAULT_VIDEO,
                 '-f', 'image2pipe',
                 '-pix_fmt', 'rgb24',
                 '-vcodec', 'rawvideo', '-',
                 '-f', 'sdl', 'original face'
                 ]
input_pipe = sp.Popen(input_command, stdout=sp.PIPE, bufsize=10 ** 8)

output_command = [FFMPEG_BIN,
                  '-f', 'rawvideo',
                  '-vcodec', 'rawvideo',
                  '-s', '2048x2048',
                  '-pix_fmt', 'rgb24',
                  '-r', '24',
                  '-i', '-',  # input comes from a pipe
                  '-an',  # no audio
                  '-f', 'v4l2', '/dev/video0'
                  # '-f', 'sdl', 'protected face'
                  ]

output_pipe = sp.Popen(output_command, stdin=sp.PIPE, stderr=sp.PIPE)

dashcast_cmd = 'cd /home/tangbao/codes/vimeo-depth-player/assets/dash; \
                DashCast -vf video4linux2 -v4l2f rawvideo -vfr 24 -vres 2048x2048 -v /dev/video0 -live -seg-dur 1000 \
                -time-shift 600 -mpd manifest.mpd;'


def get_args():
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--description", type=str, default="LiVSec System")
    parser.add_argument("--data_path", type=str, default="/data/faceid")

    # MODULE specific args
    parser = AttackFaceIDModule.add_model_specific_args(parser)

    # DATA specific args
    parser = AttackFaceIDDataModuleLiVSec.add_data_specific_args(parser)

    # TRAINER args
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--no_workers", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args([])

    return args


def get_models(faceid_model='modules/livsec_face_auth.ckpt',
               attack_faceid_model='modules/livsec_protect.ckpt'
               ):
    faceid = FaceIDModule.load_from_checkpoint(faceid_model)
    faceid.freeze()
    faceid.cuda()

    attack_faceid = AttackFaceIDModule.load_from_checkpoint(attack_faceid_model)
    attack_faceid.freeze()
    attack_faceid.cuda()
    return faceid, attack_faceid


def get_dataloader(args):
    dataset = AttackFaceIDDatasetLiVSec(args, split='livsec')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return iter(dataloader)


if __name__ == '__main__':
    seed_everything(1)
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    faceid, attack_faceid = get_models()
    dataloader = get_dataloader(args)
    cosine = torch.nn.CosineSimilarity()
    ssim_calc = SSIM(data_range=1, size_average=True)
    lpips_calc = lpips.LPIPS(net='alex').cuda()

    raw_similarity_list = []
    adv_similarity_list = []
    l2_norm_list = []
    ssim_list = []
    lpips_list = []

    REUSE_PERT = None
    REUSE_PERT_IDX = None
    REUSE = False
    start_time = time.perf_counter_ns()
    i = 0
    while i <= 120:  # 120 for default video
        frame_start_time = time.perf_counter_ns()
        i = i + 1
        print('=========================' + str(i) + '=========================')
        a = time.perf_counter_ns()
        raw_frame = input_pipe.stdout.read(2048 * 2048 * 3)
        frame = np.fromstring(raw_frame, dtype='uint8')
        video_not_end = frame.size == 2048 * 2048 * 3

        if video_not_end:
            frame = frame.reshape((2048, 2048, 3))
            input_pipe.stdout.flush()
            b = time.perf_counter_ns()
            print('read frame',  b - a)
            a = time.perf_counter_ns()
            h, w, d_face_s, d_face_v = get_face_rgbd(args, i-1, frame)
            print('get face rgbd', time.perf_counter_ns()-a)

            if REUSE:
                if i - REUSE_PERT_IDX == REUSE_FREQ:
                    REUSE = False

            dataload_new_start = time.perf_counter_ns()
            x_ref = torch.load(os.path.join(args.data_path, 'livsec', "person0_pose0.pt")).unsqueeze(0).cuda()
            x = torch.load(os.path.join(args.data_path, 'livsec', "person0_pose" + str(i-1) + ".pt")).unsqueeze(0).cuda()
            x_mask = (x[0][3] < 0.9999).unsqueeze(0).cuda()
            print('dataload time new', time.perf_counter_ns() - dataload_new_start)

            with torch.no_grad():
                if not REUSE:
                    aa = time.perf_counter_ns()
                    x_adv = attack_faceid(x, x_mask)
                    print('attack', time.perf_counter_ns() - aa)
                    REUSE = True
                    REUSE_PERT = x_adv - x
                    REUSE_PERT_IDX = i

                else:
                    x_adv = x + REUSE_PERT

                if MODE != STREAMING_MODE:
                    embed_x_ref = faceid(x_ref)
                    embed_x = faceid(x)
                    embed_x_adv = faceid(x_adv)

                    cosine_x = round(cosine(embed_x_ref, embed_x).item(), 3)
                    cosine_x_adv = round(cosine(embed_x_ref, embed_x_adv).item(), 3)
                    print('similarity score', cosine_x, cosine_x_adv)
                    raw_similarity_list.append(cosine_x)
                    adv_similarity_list.append(cosine_x_adv)
                    ssim = round(ssim_calc(unnormalize(x_adv)[:,:3,:,:], unnormalize(x)[:,:3,:,:]).item(), 3)
                    lpips = round(lpips_calc.forward(unnormalize(x_adv)[:, :3, :, :], unnormalize(x)[:, :3, :, :]).item(), 4)

                    print('ssim', ssim)
                    ssim_list.append(ssim)
                    print('lpips', lpips)
                    lpips_list.append(lpips)

                    x = to_numpy(x)
                    x_adv = to_numpy(x_adv)
                    x_ref = to_numpy(x_ref)
                    noise = abs(x_adv - x)
                    l2_nrom = np.linalg.norm(noise) / np.linalg.norm(x)
                    print(l2_nrom)
                    l2_norm_list.append(l2_nrom)

                    # if (i-1) % 10 == 1:
                    #     display(x_ref, x, x_adv, cosine_x, cosine_x_adv, l2_nrom, ssim, i-1, save=True)

            if MODE == STREAMING_MODE:
                a = time.perf_counter_ns()
                x_adv = to_numpy(x_adv)
                print('x adv to numpy', time.perf_counter_ns() - a)

            face_recover_start = time.perf_counter_ns()
            rgb_face = np.rot90((x_adv[:, :, :3] * 255).astype(np.uint8), -1)
            frame[h:h + args.img_size:, w:w + args.img_size, :] = rgb_face

            # recover depth face
            d_face_hsv = np.zeros((args.img_size, args.img_size, 3)).astype(np.uint8)
            d_face_hsv[:, :, 0] = np.rot90(x_adv[:, :, 3], -1) * 180.
            d_face_hsv[:, :, 1] = d_face_s
            d_face_hsv[:, :, 2] = d_face_v
            d_face = cv2.cvtColor(d_face_hsv, cv2.COLOR_HSV2RGB)

            half_h = int(frame.shape[0] / 2)
            frame[h+half_h:h+args.img_size+half_h, w:w+args.img_size, :] = d_face

            print('face recover time', time.perf_counter_ns() - face_recover_start)

            output_start = time.perf_counter_ns()
            output_pipe.stdin.write(frame.tobytes())
            print('output to pipe', time.perf_counter_ns() - output_start)

            print('frame end time', time.perf_counter_ns() - frame_start_time)

            if i == 1 and MODE == STREAMING_MODE:
                os.system("gnome-terminal -t dashcast_win -e 'bash -c \"" + dashcast_cmd + " bash\" '")

    print('=====================Summary===============')
    print('total time', time.perf_counter_ns() - start_time)
    if MODE != STREAMING_MODE:
        ave_raw_similarity = round(sum(raw_similarity_list) / len(raw_similarity_list), 3)
        ave_adv_similarity = round(sum(adv_similarity_list) / len(adv_similarity_list), 3)
        ave_l2 = round(sum(l2_norm_list) / len(l2_norm_list), 4)
        ave_lpips = round(sum(lpips_list) / len(lpips_list), 4)
        ave_ssim = round(sum(ssim_list) / len(ssim_list), 4)

        success_num = 0
        valid_case = 0
        for i in range(len(raw_similarity_list)):
            if raw_similarity_list[i] >= 0.9:
                valid_case += 1
                if adv_similarity_list[i] < 0.9:
                    success_num += 1

        success_rate = round(success_num / valid_case, 4)

        print('average raw similarity', ave_raw_similarity)
        print('average adv similarity', ave_adv_similarity)
        print('success rate', success_rate, str(success_num) + '/' + str(valid_case))
        print('average l2 norm', ave_l2)
        print('average ssim', ave_ssim)
        print('average lpips', ave_lpips)

        print('l2 min', np.min(l2_norm_list))
        print('l2 first quartile', np.quantile(l2_norm_list, 0.25))
        print('l2 medium', np.median(l2_norm_list))
        print('l2 3rd qurtile', np.quantile(l2_norm_list, 0.75))
        print('l2 max', np.max(l2_norm_list))
        print('l2 ave', np.mean(l2_norm_list))

        print('ssim min', np.min(ssim_list))
        print('ssim first quartile', np.quantile(ssim_list, 0.25))
        print('ssim medium', np.median(ssim_list))
        print('ssim 3rd qurtile', np.quantile(ssim_list, 0.75))
        print('ssim max', np.max(ssim_list))
        print('ssim ave', np.mean(ssim_list))

        print('lpips min', np.min(lpips_list))
        print('lpips first quartile', np.quantile(lpips_list, 0.25))
        print('lpips medium', np.median(lpips_list))
        print('lpips 3rd qurtile', np.quantile(lpips_list, 0.75))
        print('lpips max', np.max(lpips_list))
        print('lpips ave', np.mean(lpips_list))

        plt.hist(raw_similarity_list, range=[0, 1], bins=100)
        plt.hist(adv_similarity_list, range=[0, 1], bins=100)
        plt.title('Dataset #2. Similarity Score Distribution w/ and w/o $\it{LiVSec}$.')
        plt.legend(['Similarity Score w/o $\it{LiVSec}$', 'Similarity Score w/ $\it{LiVSec}$'], fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Similarity Score", fontsize=14)
        plt.ylabel("Number of Cases", fontsize=14)
        plt.show()


