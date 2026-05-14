AUDIT
    pip install opencv-python pillow numpy imagehash matplotlib scipy
    python audit.py --input_dir ./UTD-CV-Data/
RESIZE
    python resize.py --input_dir ./UTD-CV-Data/  --output_dir /Depth-Anything-V2/UTD_cust/processed/images
ANALYZE
    python analyze.py --metadata ./processed/metadata.json --audit_report ./processed/audit_report.json
    #futureTODO: swap audit_report path to original
    #review info.md for information on what datasets_analysis.png actually means and why its important.

    python gen.py --image_dir ./UTD_cust/processed/images --encoder vitl --visualize --weights /checkpoints/depth_anything_v2_vitl.pth


SPLIT
    python splits.py --processed_dir /Depth-Anything-V2/UTD_cust/processed/

check dataloader
        dataset/
        ├── depth/
        ├── images/
        └── masks/
    python UTD_dataloader.py --processed_dir UTD_cust/

TRAIN

    python train.py --processed_dir UTD_cust/ --strategy decoder_only --losses silog edge ssim --epochs 400 --batch_size 16 --lr 1e-5
        #checkpointing and logs files ened to be named dynamically from args for tracking 
        ##python train.py --processed_dir UTD_cust/ --strategy decoder_only --losses silog edge ssim --loss_weights 1.0 0.3 0.1 --epochs 250 --batch_size 8 --lr 1e-5
    python plot_training.py --log /path/to/checkpoints/training_log.json
        #path is weird need to move the file and do path checking on args
    python plot_training.py --log Depth-Anything-V2/UTD_cust/checkpoints/training_log_batch16_stratdecoder_only_encvits_epoch400.json


ABLATION test                          <name> :: <path/filename.pth> :: <path_dataset>
 python ablation.py --experiments     depth_anything_v2_vits::checkpoints/depth_anything_v2_vits.pth::./UTD_cust     depth_anything_v2_vits_ft_L105025::./UTD_cust/checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_05_025.pt::./UTD_cust depth_anything_v2_vits_ft_L10301::./UTD_cust/checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch400_L_1_03_01.pt::./UTD_cust depth_anything_v2_vits_ft_L10301_e150::./UTD_cust/checkpoints/best_model_batch8_stratdecoder_only_encvits_epoch150_L_1_03_01.pt::./UTD_cust

Export
