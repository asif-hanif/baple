import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# import datasets.oxford_pets
# import datasets.oxford_flowers
# import datasets.fgvc_aircraft
# import datasets.dtd
# import datasets.eurosat
# import datasets.stanford_cars
# import datasets.food101
# import datasets.sun397
# import datasets.caltech101
# import datasets.ucf101
# import datasets.imagenet
# import datasets.imagenet_sketch
# import datasets.imagenetv2
# import datasets.imagenet_a
# import datasets.imagenet_r

import datasets.kather
import datasets.pannuke
import datasets.wsss4luad
import datasets.digestpath
import datasets.covid
import datasets.rsna18
import datasets.mimic

import trainers.coop
import trainers.zsclip


import sys, os
medclip_path = os.path.join(os.getcwd(),"MedCLIP")
sys.path.insert(0, medclip_path)

biomedclip_path = os.path.join(os.getcwd(),"OpenCLIP", "src")
sys.path.insert(0, biomedclip_path)

import trainers.coop_medclip
import trainers.coop_biomedclip
import json

from utils import save_results


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    cfg.BACKDOOR = CN()
    cfg.BACKDOOR.POISON_PERCENTAGE= 5
    cfg.BACKDOOR.TARGET_CLASS= 0
    cfg.BACKDOOR.NOISE_EPS= 8


    cfg.BACKDOOR.PATCH_TYPE = "text" # "text", "random-patch", "image-patch"
    cfg.BACKDOOR.POSITION= "bottom-left" # "top-left", "top-center", "top-right", "center-left", "center-center", "center-right", "bottom-left", "bottom-center", "bottom-right" 
    cfg.BACKDOOR.TRIGGER_SIZE= 24 
    cfg.BACKDOOR.TRIGGER_IMG_PATH= "<PATH>"
    
    
    cfg.MODEL_NAME = CN()
    cfg.MODEL_NAME = ""
    cfg.MODEL_ROOT = CN()
    cfg.MODEL_ROOT = ""
    cfg.DATASET_NAME = CN()
    cfg.DATASET_NAME = ""

    cfg.DEVICE = CN()
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.DTYPE = CN()
    cfg.DTYPE = "float32"


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # 3. From input arguments
    reset_cfg(cfg, args)
    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg



def main(args):
    cfg = setup_cfg(args)
    
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))
    
    
    trainer = build_trainer(cfg)

    # self.val_loader.dataset.trigger = self.train_loader_x.dataset.trigger
    # self.test_loader.dataset.trigger = self.train_loader_x.dataset.trigger

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        breakpoint()
        trainer.test_loader.dataset.backdoor_tags = torch.zeros(7180)
        trainer.test()
        return

    if not args.no_train:

        results = trainer.train()

        print(f"\n\n\n\nRESULTS  (MODEL: {cfg.MODEL_NAME.upper()}   DATASET: {cfg.DATASET_NAME.upper()}   TARGET CLASS: {cfg.BACKDOOR.TARGET_CLASS})\n")
        print("#############################################################################")
        print("-------------------------------------   -------------------------------------")
        print("                 CLEAN                                 BACKDOOR           ")
        print("-------------------------------------   -------------------------------------")
        print(f"Accuracy = {results[0]['accuracy']/100:0.3f}    F-1 Score = {results[0]['macro_f1']/100:0.3f}   Accuracy = {results[1]['accuracy']/100:0.3f}    F-1 Score = {results[1]['macro_f1']/100:0.3f}")
        print("-------------------------------------   -------------------------------------")
        print("\n#############################################################################")
        print("\n\n")
        

        var_name = "target_class"
        var_value = str(cfg.BACKDOOR.TARGET_CLASS)
        save_results(cfg, results, var_name, var_value)
            
        return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument( "--resume", type=str, default="", help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=-1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file",type=str,default="",help="path to config file for dataset setup",)
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir",type=str,default="",help="load model from this directory for eval-only mode",)
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER,help="modify config options using the command-line",)
    args = parser.parse_args()

    main(args)
