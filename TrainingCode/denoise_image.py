# denoise_images.py
import os
import numpy as np
import torch
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

def denoise_images(args, autoencoder, dncnn):
    for set_cur in args.set_names:
        set_dir = os.path.join(args.set_dir, set_cur)
        if not os.path.exists(set_dir):
            print(f'Set directory not found: {set_dir}')
            continue

        result_set_dir = os.path.join(args.result_dir, set_cur)
        os.makedirs(result_set_dir, exist_ok=True)
        print(f'Saving results to: {result_set_dir}')

        psnrs = []
        ssims = []

        for im in os.listdir(set_dir):
            if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
                # 读取图像并添加噪声
                x = np.array(imread(os.path.join(set_dir, im)), dtype=np.float32) / 255.0
                np.random.seed(seed=0)  # for reproducibility
                y = x + np.random.normal(0, args.sigma / 255.0, x.shape)  # Add Gaussian noise without clipping
                y = y.astype(np.float32)
                y_ = torch.from_numpy(y).view(1, 1, y.shape[0], y.shape[1])

                if torch.cuda.is_available():
                    y_ = y_.cuda()

                # 使用 AutoEncoder 进行初步去噪
                with torch.no_grad():
                    x_autoencoded = autoencoder(y_)
                    x_autoencoded = x_autoencoded.view(y.shape[0], y.shape[1])
                    x_autoencoded = x_autoencoded.cpu().numpy().astype(np.float32)

                # 使用 DnCNN 进行进一步去噪
                with torch.no_grad():
                    x_dncnn = dncnn(torch.from_numpy(x_autoencoded).view(1, 1, x_autoencoded.shape[0], x_autoencoded.shape[1]).cuda())
                    x_dncnn = x_dncnn.view(x_autoencoded.shape[0], x_autoencoded.shape[1])
                    x_dncnn = x_dncnn.cpu().numpy().astype(np.float32)

                psnr_x_ = compare_psnr(x, x_dncnn, data_range=1.0)
                ssim_x_ = compare_ssim(x, x_dncnn, data_range=1.0)
                psnrs.append(psnr_x_)
                ssims.append(ssim_x_)

                if args.save_result:
                    name, ext = os.path.splitext(im)
                    result_path = os.path.join(result_set_dir, name + '_denoised' + ext)
                    imsave(result_path, np.clip(x_dncnn, 0, 1))  # save the denoised image
                    print(f'Saved denoised image: {result_path}')

        psnr_avg = np.mean(psnrs)
        ssim_avg = np.mean(ssims)
        print(f'Dataset: {set_cur} \n  PSNR = {psnr_avg:.2f} dB, SSIM = {ssim_avg:.4f}')

        if args.save_result:
            results_path = os.path.join(result_set_dir, 'results.txt')
            np.savetxt(results_path, np.hstack((psnrs, ssims)), fmt='%2.4f')
            print(f'Saved results: {results_path}')