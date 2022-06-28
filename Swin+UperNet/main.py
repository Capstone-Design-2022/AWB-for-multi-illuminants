import argparse
import os
import torch

from dataloader_v4 import get_loader
from solver import Solver

def print_config(config):
    print("="*20, "Configuration", "="*20)
    print(config)
    print("="*55)

def main(config):
    # Create directories if not exist
    print_config(config)

    train_loader = None
    valid_loader = None
    test_loader = None

    if config.mode == 'train':	
        train_loader = get_loader(config,'train')
        valid_loader = get_loader(config,'val')
    elif config.mode == 'test':
        if config.checkpoint == None:
            print("[ERROR]\tCheckpoint required in test mode!")
            exit()
        test_loader = get_loader(config,'test')

    solver = Solver(config, train_loader, valid_loader, test_loader)
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()



# def spmd_main(local_world_size, local_rank):
#     # These are the parameters used to initialize the process group
#     env_dict = {
#         key: os.environ[key]
#         for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
#     }
#     # print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
#     torch.distributed.init_process_group(backend="nccl")
#     # print(
#     #     f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
#     #     + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
#     # )

#     # Explicitly setting seed to make sure that models created in two processes
#     # start from same random weights and biases.
#     # torch.manual_seed(42)
#     # demo_basic(local_world_size, local_rank)

#     # Tear down the process group
#     # torch.distributed.destroy_process_group()
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=0) # added
    # parser.add_argument("--local_world_size", type=int, default=3) # added
    
    
    # model hyper-parameters
    parser.add_argument('--model_type', type=str, default='Swin_UperNet') #'U_Net' #UNet_3Plus #UNet_3Plus_DeepSup 
    parser.add_argument('--img_ch', type=int, default=3)
    parser.add_argument('--output_ch', type=int, default=2)
    parser.add_argument('--checkpoint', type=str, default=None)

    # training hyper-parameters
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=8) #32 #8 #16
    parser.add_argument('--lr', type=float, default=4e-4) #5e-4=0.0005 #6e-4
    parser.add_argument('--num_epochs_decay', type=int, default=800) #800
    parser.add_argument('--beta1', type=float, default=0.9) #0.5->0.9
    parser.add_argument('--beta2', type=float, default=0.999)

    # dataset & loader config
    parser.add_argument('--data_root', type=str, default='galaxy_512')
    parser.add_argument('--camera', type=str, default='galaxy')
    parser.add_argument('--image_size', type=int, default=224) #224 for Swin #256 for other models
    parser.add_argument('--image_pool', type=int, nargs='+', default=[1,2,3])
    parser.add_argument('--input_type', type=str, default='uvl', choices=['rgb','uvl'])
    parser.add_argument('--output_type', type=str, default='uv', choices=['illumination','uv','mixmap'])
    parser.add_argument('--uncalculable', type=int, default=-1)
    parser.add_argument('--mask_black', type=int, default=None)
    parser.add_argument('--mask_highlight', type=int, default=None)
    parser.add_argument('--mask_uncalculable', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=10)

    # data augmentation config
    parser.add_argument('--random_crop', type=str, default='yes', choices=['yes','no'])
    parser.add_argument('--illum_augmentation', type=str, default='yes', choices=['yes','no'])
    parser.add_argument('--sat_min', type=float, default=0.2)
    parser.add_argument('--sat_max', type=float, default=0.8)
    parser.add_argument('--val_min', type=float, default=1.0)
    parser.add_argument('--val_max', type=float, default=1.0)
    parser.add_argument('--hue_threshold', type=float, default=0.2)

    # path config
    parser.add_argument('--model_root', type=str, default='models')
    parser.add_argument('--result_root', type=str, default='results')
    parser.add_argument('--log_root', type=str, default='logs')

    # Misc
    parser.add_argument('--save_epoch', type=int, default=200,
                        help='number of epoch for auto saving, -1 for turn off')
    parser.add_argument('--multi_gpu', type=int, default=1, choices=[1,2], #1, [0,1,2]
                        help='0 for single-GPU, 1 for multi-GPU')
    parser.add_argument('--save_result', type=str, default='no')
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--val_step', type=int, default=1)
    parser.add_argument('--change_log', type=str)

    config = parser.parse_args()
    # spmd_main(config.local_world_size, config.local_rank)
    main(config)