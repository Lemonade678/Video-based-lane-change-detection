import os
import cv2
import torch
import numpy as np
from clrnet.utils.config import Config
from mmcv.parallel import MMDataParallel
from clrnet.models.registry import build_net
from clrnet.utils.visualization import imshow_lanes


def preprocess(img, img_w, img_h, cut_height):
    img_pre = img[cut_height:, :, :]
    img_pre = cv2.resize(img_pre, (img_w, img_h))
    img_pre = (img_pre / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre


def process_video(input_video, output_video, img_w, img_h, cut_height, cfg, model):
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Failed to open video: {input_video}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, fps, (orig_width, orig_height))
    if not out.isOpened():
        print("Failed to initialize VideoWriter.")
        return

    print(f"Video Resolution: {orig_width}x{orig_height}, FPS: {fps}, Total Frames: {frame_count}")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"Frame read failed at frame {frame_idx}. Ending process.")
            break

        # Step 1: Resize frame to 1640x590 before processing
        resized_frame = cv2.resize(frame, (1640, 590))

        # Step 2: Preprocess resized frame
        img_pre = preprocess(resized_frame, img_w, img_h, cut_height).to("cuda")

        with torch.no_grad():
            output = model(img_pre)
            if output is None:
                print(f"Model output is None at frame {frame_idx}. Skipping frame.")
                continue
            lanes = model.module.heads.get_lanes(output)[0]
            lanes = [lane.to_array(cfg) for lane in lanes]
            imshow_lanes(resized_frame, lanes, show=False, out_file=None)

        # Step 3: Resize processed frame back to original resolution for output
        final_frame = cv2.resize(resized_frame, (orig_width, orig_height))
        out.write(final_frame)

        frame_idx += 1
        print(f"Processed frame {frame_idx}/{frame_count}", end="\r")

    cap.release()
    out.release()
    print(f"\nOutput saved to: {output_video}")


if __name__ == "__main__":
    # 輸入影片路徑
    input_video = r"/home/dgxadmin/yoyo/CLRNet-main/inputvideos/short_ODBB.mp4"

    # 自動生成輸出影片名稱
    base_name = os.path.basename(input_video)
    name, ext = os.path.splitext(base_name)
    output_video = f"/home/dgxadmin/yoyo/CLRNet-main/outputvideos/{name}_output.mp4"

    # 圖片處理參數
    img_width = 800
    img_height = 320
    cut_height = 270

    # 加載配置與模型
    cfg = Config.fromfile(r"/home/dgxadmin/yoyo/CLRNet-main/configs/clrnet/clr_resnet18_culane.py")
    checkpoint_file_path = r"/home/dgxadmin/yoyo/CLRNet-main/0601.pth"
    net = build_net(cfg)
    net = MMDataParallel(net, device_ids=[0]).cuda()
    pretrained_model = torch.load(checkpoint_file_path)
    net.load_state_dict(pretrained_model['net'], strict=False)
    net.eval()
    model = net.to("cuda")

    # 處理影片
    process_video(input_video, output_video, img_width, img_height, cut_height, cfg, model)
