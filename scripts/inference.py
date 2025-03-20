import argparse
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import cfg
from utils.checkpoint import VSTGCheckpointer
from models import build_model, build_postprocessors
from utils.misc import NestedTensor
from utils.bounding_box import BoxList
from engine.evaluate import single_forward, linear_interp
import ffmpeg
import torchvision
import numpy as np
from utils.logger import setup_logger
import cv2

def make_bbox_image(image, bbox, labels=None, color=None):
    image_with_box = torchvision.utils.draw_bounding_boxes(
        image, torch.tensor(bbox), width=4, labels=[labels], colors=color
    )
    return image_with_box


def get_video_image_size(video_path):
    #print(f"###########{video_path}")
    probe = ffmpeg.probe(video_path)
    #print(f"###########{probe}")
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )

    if not video_stream:
        raise ValueError(f"❌ 비디오 스트림을 찾을 수 없습니다: {video_path}")
    
    width = video_stream["width"]
    height = video_stream["height"]
    
    # ✅ `nb_frames`가 있으면 사용, 없으면 `duration * FPS`로 계산
    if "nb_frames" in video_stream:
        total_frames = int(video_stream["nb_frames"])
    else:
        duration = float(video_stream.get("duration", 0))  # 없으면 기본값 0
        frame_rate = eval(video_stream.get("r_frame_rate", "30/1"))  # FPS 기본값 30
        total_frames = int(duration * frame_rate)  # ✅ 직접 계산

    return width, height, total_frames


def get_video_np_array(video_path):
    out, _ = (
        ffmpeg.input(video_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True)
    )

    return np.frombuffer(out, np.uint8)


def load_video(video_path, input_resolution, device, num_frames=128):
    W, H, total_frames = get_video_image_size(video_path)
    video_array = get_video_np_array(video_path).reshape([-1, H, W, 3])

    frame_ids = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    video_array = video_array[frame_ids]  # sampling
    video_tensor = torch.from_numpy(video_array).permute(0, 3, 1, 2).float() / 255.0

    Resize = torchvision.transforms.Resize(input_resolution, antialias=True)
    torchvision.transforms.ToTensor()
    video_tensor = Resize(video_tensor)

    masks = torch.zeros(
        num_frames, video_tensor.shape[2], video_tensor.shape[3], dtype=torch.bool
    ).to(device)
    durations = video_tensor.shape[0]

    return (
        NestedTensor(video_tensor.to(device), masks.to(device), durations=[durations]),
        (H, W),
        (video_tensor.shape[2], video_tensor.shape[3]),
        frame_ids,
    )


def load_text(text):
    return [text]


def load_target(video, original_size, input_size, frame_ids, device):
    duration = video.durations[0]

    targets = [
        {
            "item_id": 1,
            "actioness": torch.ones(duration, 1).to(device),
            "boxs": BoxList(
                torch.tensor([[0.8585, 0.3646, 0.2537, 0.1917]]).to(device),
                (input_size[1], input_size[0]),
                "xyxy",
            ),
            "durations": duration,
            "ori_size": (original_size[0], original_size[1]),
            "frame_ids": frame_ids,
            "eval": True,
            "qtype": "inter",
        }
    ]

    return targets


def inference(video_path, text, weight_path):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.merge_from_file("/workspace/gseo/CG-STVG/experiments/hcstvg2.yaml")
    cfg["INPUT"]["RESOLUTION"] = 420

    model, _, _ = build_model(cfg)

    model.to(device)
    logger = setup_logger("Inference", cfg.OUTPUT_DIR, 0)
    checkpointer = VSTGCheckpointer(cfg, model, logger=logger, is_train=False)
    _ = checkpointer.load(weight_path, with_optim=False)

    postprocessor = build_postprocessors()

    videos, original_size, input_size, frame_ids = load_video(
        video_path, cfg.INPUT.RESOLUTION, device
    )

    texts = load_text(text)
    targets = load_target(videos, original_size, input_size, frame_ids, device)

    videos1 = videos.subsample(2, start_idx=0)
    targets1 = [
        {
            "item_id": target["item_id"],
            "ori_size": target["ori_size"],
            "qtype": target["qtype"],
            "frame_ids": target["frame_ids"][0::2],
            "boxs": target["boxs"].bbox.clone(),
            "actioness": target["actioness"][0::2],
            "eval": True,
        }
        for target in targets
    ]

    videos2 = videos.subsample(2, start_idx=1)
    targets2 = [
        {
            "item_id": target["item_id"],
            "ori_size": target["ori_size"],
            "qtype": target["qtype"],
            "frame_ids": target["frame_ids"][1::2],
            "boxs": target["boxs"].bbox.clone(),
            "actioness": target["actioness"][1::2],
            "eval": True,
        }
        for target in targets
    ]

    if torch.where(targets[0]["actioness"])[0][0] % 2 == 0:
        targets1[0]["boxs"] = targets1[0]["boxs"][0::2]
        targets2[0]["boxs"] = targets2[0]["boxs"][1::2]
    else:
        targets1[0]["boxs"] = targets1[0]["boxs"][1::2]
        targets2[0]["boxs"] = targets2[0]["boxs"][0::2]

    bbox_pred1, temp_pred1 = single_forward(
        cfg, model, videos1, texts, targets1, device, postprocessor
    )
    bbox_pred2, temp_pred2 = single_forward(
        cfg, model, videos2, texts, targets2, device, postprocessor
    )

    bbox_pred, temp_pred = {}, {}
    for vid in bbox_pred1:
        bbox_pred1[vid].update(bbox_pred2[vid])
        bbox_pred[vid] = linear_interp(bbox_pred1[vid])
        temp_pred[vid] = {
            "sted": [
                min(temp_pred1[vid]["sted"][0], temp_pred2[vid]["sted"][0]),
                max(temp_pred1[vid]["sted"][1], temp_pred2[vid]["sted"][1]),
            ]
        }
        if "qtype" in temp_pred1[vid]:
            temp_pred[vid]["qtype"] = temp_pred1[vid]["qtype"]

    item_id = targets[0]["item_id"]
    bbox_pred = bbox_pred[item_id]
    temp_pred = temp_pred[item_id]["sted"]

    return bbox_pred, temp_pred


def make_inference_video(video_path, bbox_pred, temp_pred, save_path, text):
    font_scale = 0.3
    font_thickness = 1

    W, H, total_frames = get_video_image_size(video_path)
    video_array = get_video_np_array(video_path).reshape([-1, H, W, 3])
    video_tensor = torch.from_numpy(video_array.copy()).permute(0, 3, 1, 2).float()
    video_tensor = video_tensor / 255.0

    process = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{W}x{H}", framerate=30
        )
        .output(save_path, pix_fmt="yuv420p", vcodec="libx264", preset="ultrafast", crf=23)
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    for frame in range(total_frames):
        img = video_tensor[frame, :, :, :]
        img = torchvision.transforms.functional.convert_image_dtype(
            img, dtype=torch.uint8
        )

        # 🔥 NumPy 배열로 변환
        frame_np = img.permute(1, 2, 0).numpy().astype(np.uint8)

        # 🔥 좌상단에 텍스트 배경 (흰색 사각형) 추가
        text_position = (5, 15)  # (x, y) 좌표
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        cv2.rectangle(frame_np, (text_position[0] - 4, text_position[1] - 10), 
                      (text_position[0] + text_size[0] + 4, text_position[1] + 10), 
                      (255, 255, 255), -1)  # 흰색 배경

        # 🔥 좌상단에 까만색 텍스트 추가
        cv2.putText(frame_np, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        # 🔥 NumPy 배열을 다시 Torch 텐서로 변환
        img = torch.tensor(frame_np).permute(2, 0, 1)

        # 🔥 Bounding Box 추가
        if frame in range(*temp_pred):
            bbox = bbox_pred.get(frame, [])  # 🔥 bbox_pred에서 frame 키가 없을 경우 예외 방지
            img = make_bbox_image(img, bbox, "", "blue")  # 🔥 Predict 대신 빈 문자열

        frame = img.permute(1, 2, 0).numpy().astype(np.uint8)  # 다시 NumPy로 변환
        try:
            process.stdin.write(frame.tobytes())  # 🔥 예외 처리 추가
        except BrokenPipeError:
            print(f"inference 안된 비디오: {video_path}")
            break

    process.stdin.close()
    process.wait()
    #print(f"Inference saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Training")
    for arg in ["--video", "--text", "--weights", "--save-path"]:
        parser.add_argument(arg, type=str)
    args = parser.parse_args()

    bbox_pred, temp_pred = inference(args.video, args.text, args.weights)
    make_inference_video(args.video, bbox_pred, temp_pred, args.save_path)


if __name__ == "__main__":

    # python3 infer.py --video="../videos/test_video.mp4" --text="What is the cat in the kitchen trying to eat from the person’s hand?" --weights="../weights/vidstg.pth" --save-path="result.mp4"

    main()
