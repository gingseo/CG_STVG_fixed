import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
import datetime
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from utils.comm import synchronize, get_rank, is_main_process, reduce_loss_dict
from utils.logger import setup_logger
from utils.misc import mkdir, save_config, set_seed, to_device
from utils.checkpoint import VSTGCheckpointer
from datasets import make_data_loader, build_evaluator, build_dataset
from models import build_model, build_postprocessors
from engine import make_optimizer, adjust_learning_rate, update_ema, do_eval
from utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter
from inference import inference, make_inference_video
import json
import matplotlib.pyplot as plt
memory_usage_history = {"iteration": [], "memory": []}
def save_memory_usage_graph(save_path):

    # GPU 메모리 사용량을 그래프로 저장
    # - save_path: 저장할 폴더 경로

    if not memory_usage_history["iteration"]:
        return  # 기록된 데이터가 없으면 함수 종료

    plt.figure(figsize=(8, 5))
    plt.plot(memory_usage_history["iteration"], memory_usage_history["memory"], 
             marker="o", linestyle="-", color="green", label="Memory Usage (MB)")
    
    plt.xlabel("Iteration")
    plt.ylabel("Memory Usage (MB)")
    plt.title("GPU Memory Usage Over Iterations")
    plt.legend()
    plt.grid(True)

    # 저장 경로 설정
    os.makedirs(save_path, exist_ok=True)
    graph_path = os.path.join(save_path, "memory_usage.png")

    # 그래프 저장
    plt.savefig(graph_path)
    plt.close()

def train(cfg, local_rank, distributed, logger):
    #print(f"🔥 Training on GPU {torch.cuda.current_device()} (local_rank={local_rank})")
    logger.info(f"🔥 Training on GPU {torch.cuda.current_device()} (local_rank={local_rank})")
    model, criteria, weight_dict = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    criteria.to(device)

    optimizer = make_optimizer(cfg, model, logger)
    model_ema = deepcopy(model) if cfg.MODEL.EMA else None
    model_without_ddp = model
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR
    infer_dir = cfg.INFER_DIR
    save_to_disk = local_rank == 0
    checkpointer = VSTGCheckpointer(
        cfg, model_without_ddp, model_ema, optimizer, output_dir, save_to_disk, logger, is_train=True
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    arguments.update(extra_checkpoint_data)
    
    verbose_loss = set(["loss_bbox", "loss_giou", "loss_sted", "loss_conf"])
    
    if cfg.SOLVER.USE_ATTN:
        verbose_loss.add("loss_guided_attn")
    
    if cfg.MODEL.CG.USE_ACTION:
        verbose_loss.add("loss_actioness")
    
    # Prepare the dataset cache
    if local_rank == 0:
        split = ['train', 'test']
        if cfg.DATASET.NAME == "VidSTG":
            split += ['val']
        for mode in split:
            _ = build_dataset(cfg, split=mode, transforms=None)
       
    synchronize()

    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loader = make_data_loader(
        cfg,
        mode='val' if cfg.DATASET.NAME == "VidSTG" else "test",
        is_distributed=distributed,
    )
    test_sub_data_loader = make_data_loader(
        cfg, 
        mode='test_sub', 
        is_distributed=distributed, 
    )

    if cfg.TENSORBOARD_DIR and is_main_process():
        writer = SummaryWriter(cfg.TENSORBOARD_DIR)
    else:
        writer = None
    
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    validation_period = 3000 #cfg.SOLVER.VAL_PERIOD
    logger.info("Start training")

    if cfg.SOLVER.PRE_VAL:
        logger.info("Validating before training")
        run_eval(cfg, model, model_ema, logger, val_data_loader, device)
    
    metric_logger = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    for iteration, batch_dict in enumerate(train_data_loader, start_iter):
        model.train()
        criteria.train()

        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        videos = batch_dict['videos'].to(device)
        texts = batch_dict['texts']
        #video_name = batch_dict['video_names']
        video_shape = list(videos.tensors.shape) if hasattr(videos, "tensors") else list(videos.shape)  
        current_memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        durations = batch_dict['durations']
        targets = to_device(batch_dict["targets"], device) 
        targets[0]["durations"] = durations
        if iteration % 1 == 0:
            logger.info(
                f"🎥 Iter: {iteration}/{max_iter}: , "#Video - {video_name}, "
                f"GPU: {local_rank} | "
                f"{video_shape} | "
                f"Mem Usage: {current_memory:.2f} MB | "
                f"Iter Time: {data_time:.4f}s"
            )
        outputs = model(videos, texts, targets, iteration/max_iter)

        # compute loss
        loss_dict = criteria(outputs, targets, durations)

        # loss used for update param
        # assert set(weight_dict.keys()) == set(loss_dict.keys())
        losses = sum(loss_dict[k] * weight_dict[k] for k in \
                            loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled" : v \
                        for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k : v * weight_dict[k] for k, v in loss_dict_reduced.items()\
                 if k in weight_dict and k in verbose_loss
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        # filter unrelated loss
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)

        optimizer.zero_grad()
        losses.backward()
        if cfg.SOLVER.MAX_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.SOLVER.MAX_GRAD_NORM)
        optimizer.step()

        adjust_learning_rate(cfg, optimizer, iteration, max_iter)

        if model_ema is not None:
            update_ema(model, model_ema, cfg.MODEL.EMA_DECAY)

        batch_time = time.time() - end
        end = time.time()
        metric_logger.update(time=batch_time, data=data_time)

        eta_seconds = metric_logger.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if writer is not None and is_main_process() and iteration % 50 == 0:
            for k in loss_dict_reduced_scaled:
                writer.add_scalar(f"{k}", metric_logger.meters[k].avg, iteration)

        if iteration % 100 == 0:
            save_memory_usage_graph(cfg.OUTPUT_DIR)

        if iteration % 1000 == 0:
            elapsed_time = time.time() - start_training_time
            elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            logger.info(f"⏳ Training Progress: {iteration}/{max_iter} iterations completed. Elapsed time: {elapsed_time_str}")

        if iteration % 50 == 0 or iteration == max_iter:
            logger.info(
                metric_logger.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter} / {max_iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "lr_vis_encoder: {lr_vis:.6f}",
                        "lr_text_encoder: {lr_text:.6f}",
                        "lr_temp_decoder: {lr_temp:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    max_iter = max_iter,
                    meters=str(metric_logger),
                    lr=optimizer.param_groups[0]["lr"],
                    lr_vis=optimizer.param_groups[1]["lr"],
                    lr_text=optimizer.param_groups[2]["lr"],
                    lr_temp=optimizer.param_groups[3]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
            
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        if cfg.SOLVER.TO_VAL and iteration % validation_period == 0:
            # 최신 weight 저장 후 inference 실행
            latest_weight_path = os.path.join(output_dir, f"latest_model.pth")
            checkpointer.save(f"latest_model", **arguments)

            for idx, test_sample in enumerate(test_sub_data_loader.dataset.all_gt_data[:-1]):
                video_filename = test_sample["vid"] 
                video_path = os.path.join(cfg.DATA_DIR, "v2_video", video_filename) 
                text = test_sample["description"]
                save_path = os.path.join(infer_dir, f"{idx}_{iteration}.mp4")

                bbox_pred, temp_pred = inference(video_path, text, latest_weight_path)
                make_inference_video(video_path, bbox_pred, temp_pred, save_path, text)

            save_path = infer_dir
            run_eval(cfg, model, model_ema, logger, val_data_loader, test_sub_data_loader, device, writer, iteration, save_path)

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    if writer is not None:
        writer.close()

    return model, model_ema


def run_eval(cfg, model, model_ema, logger, val_data_loader, test_sub_data_loader,  device, writer=None, iteration=None, save_path=None):
    logger.info("Start validating")
    test_model = model_ema if model_ema is not None else model
    evaluator = build_evaluator(cfg, logger, mode='val' if cfg.DATASET.NAME == "VidSTG" else "test")
    postprocessor = build_postprocessors()
    torch.cuda.empty_cache()
    
    results = do_eval(
        cfg,
        mode='val',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_data_loader,
        evaluator=evaluator,
        device=device
    )
    result_sub = do_eval(
        cfg,
        mode='val',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=test_sub_data_loader,
        evaluator=evaluator,
        device=device
    )
    synchronize()
    
    # Logger 출력
    logger.info("👀👀👀👀👀👀👀👀👀👀👀👀👀👀👀")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # TensorBoard 기록
    if writer is not None and is_main_process() and iteration is not None:
        for metric, value in results.items():
            writer.add_scalar(f"eval/{metric}", value, iteration)
    
    #  개별 비디오 성능 가져오기
    if save_path is not None and is_main_process():
        vid_metrics = result_sub.get("vid_metrics", {})  #  개별 비디오 성능 가져오기
        for video_id, metrics in vid_metrics.items():
            save_video_metrics_graph(metrics, video_id, save_path, iteration)
    
    return results

def save_video_metrics_graph(metrics, video_id, save_path, iteration):
    """
    각 비디오(video_id)의 학습 에폭(iteration)별 성능 변화를 그래프로 저장하는 함수
    
    Args:
        metrics (dict): 비디오 성능 지표 ("yiou", "tiou" 값 리스트)
        video_id (str): 비디오 ID
        save_path (str): 그래프를 저장할 경로
        iteration (int): 현재 에폭
    """
    
    # 저장 경로 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 기존 데이터가 있으면 불러오기 (누적 저장을 위해)
    # 저장 파일명 설정
    graph_path = os.path.join(save_path, f"metric_{video_id}.png")
    metrics_file = os.path.join(save_path, f"metric_{video_id}.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            saved_metrics = json.load(f)
    else:
        saved_metrics = {"iteration": [], "yiou": [], "tiou": []}
    
    # 새로운 데이터 추가
    saved_metrics["iteration"].append(iteration)
    saved_metrics["yiou"].append(metrics.get("yiou", 0))  # 기본값 0
    saved_metrics["tiou"].append(metrics.get("tiou", 0))  # 기본값 0
    
    # 업데이트된 데이터를 다시 저장
    with open(metrics_file, "w") as f:
        json.dump(saved_metrics, f, indent=4)
    
    # 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(saved_metrics["iteration"], saved_metrics["yiou"], marker='o', linestyle='-', label="yiou", color='blue')
    plt.plot(saved_metrics["iteration"], saved_metrics["tiou"], marker='s', linestyle='-', label="tiou", color='red')
    
    # 그래프 설정
    plt.xlabel("iteration")
    plt.ylabel("Score")
    plt.title(f"Performance Metrics over iterations for Video {video_id}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    # 그래프 저장
    plt.savefig(graph_path, dpi=300)
    plt.close()

def run_test(cfg, model, model_ema, logger, distributed):
    logger.info("Start Testing")
    test_model = model_ema if model_ema is not None else model
    torch.cuda.empty_cache()

    evaluator = build_evaluator(cfg, logger, mode='test')   # mode = ['val','test']
    postprocessor = build_postprocessors()
    val_data_loader = make_data_loader(cfg, mode='test', is_distributed=distributed)
    do_eval(
        cfg,
        mode='test',
        logger=logger,
        model=test_model,
        postprocessor=postprocessor,
        data_loader=val_data_loader,
        evaluator=evaluator,
        device=torch.device(cfg.MODEL.DEVICE)
    )
    synchronize()


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Grounding Training")
    parser.add_argument(
        "--config-file",
        default="/workspace/CGSTVG/experiments/hcstvg2.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local-rank", type=int, default=os.environ.get("LOCAL_RANK", -1)) #, default=0)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--use-seed",
        dest="use_seed",
        help="If use the random seed",
        default=True
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    #num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    args.distributed = num_gpus > 1

    #print(f"🔥 Running on GPU {args.local_rank} / Total GPUs: {num_gpus}")

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    if args.config_file:
        cfg.merge_from_file(args.config_file)
        
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    if args.use_seed:
        cudnn.benchmark = False
        cudnn.deterministic = True
        set_seed(args.seed + get_rank())
    
    synchronize()
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)
    logger = setup_logger("Video Grounding", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    if args.config_file:
        logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))
    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if torch.distributed.is_initialized():
        torch.distributed.barrier()  # 모든 프로세스가 준비될 때까지 기다림
    #print(f"🔥 CHECK local_rank | PID: {os.getpid()} | GPU: {torch.cuda.current_device()} | local_rank: {args.local_rank}")
    # if args.local_rank == 1:
    #     print(f"😎 {args.local_rank}, {args.distributed}")
    #     model, model_ema = train(cfg, args.local_rank, args.distributed, logger)
    #     print(f"😂 {args.local_rank}, {args.distributed}")
    # elif args.local_rank == 0:
    #     print(f"😎 {args.local_rank}, {args.distributed}")
    #     model, model_ema = train(cfg, args.local_rank, args.distributed, logger)
    # print(f"😎 {args.local_rank}, {args.distributed}")
    # if args.local_rank == 1:
    #     model, model_ema = train(cfg, args.local_rank, args.distributed, logger)
    # else:
    #     print("end")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    device = torch.cuda.current_device()
    model, model_ema = train(cfg, device, args.distributed, logger)

    # if not args.skip_test:
    #     run_test(cfg, model, model_ema, logger, args.distributed)


if __name__ == "__main__":
    main()