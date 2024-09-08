# effd
## Train
`python main.py --mode train \
--plus 1 \
--device 0 \
--batch_size 4 
--num_workers 15 
--image_size 256, 256 
--iaff_num_epochs 200 
--iaff_lr 0.00001 
--iaff_weight_decay 0.00005 
--cae_num_epochs 200 
--cae_lr 0.00001 
--cae_weight_decay 0.00005 
--levels level_2_1, level_2_2, level_3_1, level_3_2, level_3_3, level_3_4, level_4_1, level_4_2, level_4_3, level_4_4 
--pool avgpool 
--padding_mode reflect 
--gamma 4 
--alpha 3 
--betas 2, 2, 2 
--eta 8, 8 
--sigma 4, 4 
--dataset mvtec 
--categories tile, wood, cable 
--weights [8,4,1], [8,1,1], [1,4,8] 
--data_path data 
--pretrain_path pretrain 
--evaluate_interval 1`
