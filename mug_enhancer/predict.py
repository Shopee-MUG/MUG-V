import torch, os, imageio, argparse
from torchvision.transforms import v2
from torchmetrics.image import PeakSignalNoiseRatio
from torch.utils.data import Dataset, Subset
from einops import rearrange
import lightning as pl
from lightning.pytorch.callbacks import Callback, RichProgressBar
import pandas as pd
from diffsynth import ModelManager, load_state_dict, save_video
from pipelines.wan_video import WanVideoPipeline
from models.wan_video_dit import WanModel
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
import torchvision
from PIL import Image
from datetime import datetime
import time
import glob
import json


BATCH_SIZE = 2
if os.environ.get('WORLD_SIZE') is None:
    BATCH_SIZE = 1


class TextVideoDataset(Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480,
                 width=832, args=None):
        metadata = pd.read_csv(metadata_path)
        if args.use_full_video_path:
            self.path = [file_name for file_name in metadata["file_name"]]
        else:
            self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width

        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image

    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (
                num_frames - 1) * interval:
            reader.close()
            return None

        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")

        return frames

    def load_video(self, file_path):
        start_frame_id = 0
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval,
                                                self.num_frames, self.frame_process)
        return frames

    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "png", "webp"]:
            return True
        return False

    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame

    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        if self.is_image(path):
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        data = {"text": text, "video": video, "path": path}
        return data

    def __len__(self):
        return len(self.path)

class LightningModelForPredictProcess(pl.LightningModule):
    def __init__(self, dit_path, text_encoder_path, vae_path, cond_path=None, output_path='',
                 lora_rank=64, lora_alpha=4, lora_target_modules='q,k,v,o,ffn.0,ffn.2',
                 args=None):
        super().__init__()
        self.args = args
        self.output_path = output_path

        self.datetime_path = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        os.makedirs(os.path.join(output_path, self.datetime_path), exist_ok=True)

        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cuda")


        dit = WanModel(model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=1536,
                ffn_dim=8960,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=12,
                num_layers=30,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6)
        state_dict = load_state_dict(dit_path)
        missing_keys, unexpected_keys = dit.load_state_dict(state_dict, assign=True, strict=True)

        model_manager.load_models([text_encoder_path, vae_path])
        self.pipe = WanVideoPipeline.from_model_manager(model_manager, sigma_max=0.3, sigma_min=0.0)
        self.pipe.dit = dit

        if cond_path is not None:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules.split(","),
            )
            self.pipe.dit = inject_adapter_in_model(lora_config, self.pipe.dit)

            state_dict = load_state_dict(cond_path)
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k[5:]
                new_state_dict[k] = v
            missing_keys, unexpected_keys = self.pipe.load_state_dict(new_state_dict, assign=True, strict=False)
            print(f'{unexpected_keys=}')

        self.pipe.scheduler_for_predict.sigma_max = 0.3
        self.pipe.scheduler_for_predict.sigma_min = 0.0
        self.metric = PeakSignalNoiseRatio().to(self.device)

    def predict_step(self, batch, batch_idx):
        # self.metric = self.metric
        path_list = batch['path']
        batch_size = len(path_list)
        video_name_no_extn = [os.path.splitext(os.path.basename(path_list[i]))[0] for i in range(batch_size)]
        video_origin = batch['video'].to(dtype=torch.bfloat16, device=self.device)
        B, C, T, H, W = video_origin.shape

        if self.args.blur_video:
            trans1 = torchvision.transforms.Resize((int(H/2), int(W/2)), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            trans2 = torchvision.transforms.Resize((H, W), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
            video_low = [trans1(u) for u in video_origin]
            video_recover = [trans2(u) for u in video_low]
            video = torch.stack(video_recover)
        else:
            video = video_origin

        text = batch['text']

        negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",


        video_output, frames = self.pipe.enhance_video(prompt=text, negative_prompt=negative_prompt, input_video=video,
                                            height=H, width=W, num_frames=T,
                                            num_inference_steps=1, cfg_scale=args.cfg_scale, sigma_shift=1.0)

        for i in range(batch_size):
            video_filename = os.path.join(self.output_path, self.datetime_path, f'{batch_idx*batch_size+i:0>4d}_{video_name_no_extn[i]}_enhance.mp4')
            save_video(video_output[i], video_filename, fps=args.fps, quality=9)

        if self.args.blur_video:
            video_input_frame = [self.pipe.tensor2video(video[i].float()) for i in range(batch_size)]
            for i in range(batch_size):
                video_filename = os.path.join(self.output_path, self.datetime_path, f'{batch_idx*batch_size+i:0>4d}_{video_name_no_extn[i]}_input.mp4')
                save_video(video_input_frame[i], video_filename, fps=args.fps, quality=9)

        video_origin_frame = [self.pipe.tensor2video(video_origin[i].float()) for i in range(batch_size)]
        for i in range(batch_size):
            video_filename = os.path.join(self.output_path, self.datetime_path, f'{batch_idx*batch_size+i:0>4d}_{video_name_no_extn[i]}_origin.mp4')
            save_video(video_origin_frame[i], video_filename, fps=args.fps, quality=9)

        for i in range(batch_size):
            self.metric.update(frames[i], video_origin[i])
    
    def on_predict_epoch_end(self):
        psnr = self.metric.compute().item()
        print(f'{psnr=}')

        data = self.args.__dict__
        data['psnr'] = float(psnr)
        json_str = json.dumps(data, ensure_ascii=False, indent=4)

        readme_file_path = os.path.join(self.output_path, self.datetime_path, 'ckpt_path.txt')
        with open(readme_file_path, 'w', encoding='utf-8') as file:
            file.write(json_str)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a predicting script.")
    parser.add_argument(
        "--task",
        type=str,
        default="predict",
        required=True,
        choices=["predict"],
        help="Task type: `predict`.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="../pretrained_ckpt/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="../pretrained_ckpt/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="../pretrained_ckpt/MUG-V-inference/enhancer/dit_i2v.safetensors",
        help="Path of DiT.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="fps, frames per second",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="classifier free guidance",
    )
    parser.add_argument(
        "--cond_path",
        type=str,
        default="../pretrained_ckpt/MUG-V-inference/enhancer/pytorch_model.bin",
        help="cond path",
    )
    parser.add_argument(
        "--blur_video",
        default=False,
        action="store_true",
        help="blur_video when predict",
    )
    parser.add_argument(
        "--use_full_video_path",
        default=False,
        action="store_true",
        help="use_full_video_path",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="",
        help="video_path",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default=None,
        help="val_dataset_path",
    )
    args = parser.parse_args()
    return args


# 自定义回调类，记录每一步的时间
class BatchTimeCallback(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 计算本步耗时
        batch_time = time.time() - self.batch_start_time
        # 将时间记录到日志并显示在进度条
        pl_module.log("batch_time", batch_time, prog_bar=True)

def predict(args):
    dataset = TextVideoDataset(
        args.video_path,
        args.val_dataset_path,
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        args=args,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=0,
    )
    model = LightningModelForPredictProcess(
        dit_path=args.dit_path,
        text_encoder_path=args.text_encoder_path,
        vae_path=args.vae_path,
        cond_path=args.cond_path,
        output_path=args.output_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        args=args,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
        callbacks=[
            BatchTimeCallback()  # 添加自定义时间记录
        ]
    )
    trainer.predict(model, dataloader)


if __name__ == '__main__':
    args = parse_args()
    if args.task == "predict":
        predict(args)
