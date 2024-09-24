#===========================================Training networks===========================================#
# python main.py --mode train --num_domains 3 --num_outs_per_domain 5 --val_batch_size 16 \
#                --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
#                --total_iters 20000 --print_every 10 --sample_every 2000 --save_every 5000 --eval_every 5000 \
#                --train_img_dir /data/xiangyu/data/breast-ultrasound-dataset/train_dir \
#                --val_img_dir /data/xiangyu/data/breast-ultrasound-dataset/valid_dir

# python main.py --mode train --num_domains 3 --num_outs_per_domain 5 --val_batch_size 16 \
#                --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1\
#                --total_iters 50000 --print_every 20 --sample_every 5000 --save_every 5000 --eval_every 5000 \
#                --train_img_dir /data/xiangyu/data/covid19-pneumonia-dataset/train_dir \
#                --val_img_dir /data/xiangyu/data/covid19-pneumonia-dataset/valid_dir

# python main.py --mode train --num_domains 4 --num_outs_per_domain 5 --val_batch_size 16 \
#                --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1\
#                --total_iters 50000 --print_every 20 --sample_every 5000 --save_every 5000 --eval_every 5000 \
#                --train_img_dir /data/xiangyu/data/eye-disease-dataset/train_dir \
#                --val_img_dir /data/xiangyu/data/eye-disease-dataset/valid_dir

# python main.py --mode train --num_domains 6 --num_outs_per_domain 5 --val_batch_size 16 \
#                --lambda_reg 1 --lambda_sty 1 --lambda_ds 2 --lambda_cyc 1 \
#                --total_iters 8000 --print_every 10 --sample_every 500 --save_every 500 --eval_every 500 \
#                --train_img_dir /data/xiangyu/data/fetal-ultrasound-dataset/train_dir \
#                --val_img_dir /data/xiangyu/data/fetal-ultrasound-dataset/valid_dir


CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/breast-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/breast-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny1 --dataset_name breast-ultrasound-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 20000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/breast-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/breast-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny2 --dataset_name breast-ultrasound-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 20000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/breast-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/breast-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny3 --dataset_name breast-ultrasound-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 20000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/breast-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/breast-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny4 --dataset_name breast-ultrasound-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 20000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/breast-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/breast-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/breast-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny5 --dataset_name breast-ultrasound-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 20000



CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/covid19-pneumonia-dataset/train_dir \
--valid_dir /home/xiangyu/data/covid19-pneumonia-dataset/valid_dir \
--json_path /home/xiangyu/data/covid19-pneumonia-dataset/cat_to_name.json \
--save_dirname convnext_tiny1 --dataset_name covid19-pneumonia-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/covid19-pneumonia-dataset/train_dir \
--valid_dir /home/xiangyu/data/covid19-pneumonia-dataset/valid_dir \
--json_path /home/xiangyu/data/covid19-pneumonia-dataset/cat_to_name.json \
--save_dirname convnext_tiny2 --dataset_name covid19-pneumonia-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/covid19-pneumonia-dataset/train_dir \
--valid_dir /home/xiangyu/data/covid19-pneumonia-dataset/valid_dir \
--json_path /home/xiangyu/data/covid19-pneumonia-dataset/cat_to_name.json \
--save_dirname convnext_tiny3 --dataset_name covid19-pneumonia-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/covid19-pneumonia-dataset/train_dir \
--valid_dir /home/xiangyu/data/covid19-pneumonia-dataset/valid_dir \
--json_path /home/xiangyu/data/covid19-pneumonia-dataset/cat_to_name.json \
--save_dirname convnext_tiny4 --dataset_name covid19-pneumonia-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/covid19-pneumonia-dataset/train_dir \
--valid_dir /home/xiangyu/data/covid19-pneumonia-dataset/valid_dir \
--json_path /home/xiangyu/data/covid19-pneumonia-dataset/cat_to_name.json \
--save_dirname convnext_tiny5 --dataset_name covid19-pneumonia-dataset --num_classes 3 --num_domains 3 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000



CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/eye-disease-dataset/train_dir \
--valid_dir /home/xiangyu/data/eye-disease-dataset/valid_dir \
--json_path /home/xiangyu/data/eye-disease-dataset/cat_to_name.json \
--save_dirname convnext_tiny1 --dataset_name eye-disease-dataset --num_classes 4 --num_domains 4 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/eye-disease-dataset/train_dir \
--valid_dir /home/xiangyu/data/eye-disease-dataset/valid_dir \
--json_path /home/xiangyu/data/eye-disease-dataset/cat_to_name.json \
--save_dirname convnext_tiny2 --dataset_name eye-disease-dataset --num_classes 4 --num_domains 4 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/eye-disease-dataset/train_dir \
--valid_dir /home/xiangyu/data/eye-disease-dataset/valid_dir \
--json_path /home/xiangyu/data/eye-disease-dataset/cat_to_name.json \
--save_dirname convnext_tiny3 --dataset_name eye-disease-dataset --num_classes 4 --num_domains 4 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/eye-disease-dataset/train_dir \
--valid_dir /home/xiangyu/data/eye-disease-dataset/valid_dir \
--json_path /home/xiangyu/data/eye-disease-dataset/cat_to_name.json \
--save_dirname convnext_tiny4 --dataset_name eye-disease-dataset --num_classes 4 --num_domains 4 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/eye-disease-dataset/train_dir \
--valid_dir /home/xiangyu/data/eye-disease-dataset/valid_dir \
--json_path /home/xiangyu/data/eye-disease-dataset/cat_to_name.json \
--save_dirname convnext_tiny5 --dataset_name eye-disease-dataset --num_classes 4 --num_domains 4 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 50000



CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/fetal-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/fetal-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/fetal-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny1 --dataset_name fetal-ultrasound-dataset --num_classes 6 --num_domains 6 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 25000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/fetal-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/fetal-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/fetal-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny2 --dataset_name fetal-ultrasound-dataset --num_classes 6 --num_domains 6 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 25000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/fetal-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/fetal-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/fetal-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny3 --dataset_name fetal-ultrasound-dataset --num_classes 6 --num_domains 6 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 25000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/fetal-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/fetal-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/fetal-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny4 --dataset_name fetal-ultrasound-dataset --num_classes 6 --num_domains 6 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 25000

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda --train_dir /home/xiangyu/data/fetal-ultrasound-dataset/train_dir \
--valid_dir /home/xiangyu/data/fetal-ultrasound-dataset/valid_dir \
--json_path /home/xiangyu/data/fetal-ultrasound-dataset/cat_to_name.json \
--save_dirname convnext_tiny5 --dataset_name fetal-ultrasound-dataset --num_classes 6 --num_domains 6 \
--model_name convnext_tiny --train_batchSize 32 --val_batchSize 32 --resume_iter 25000