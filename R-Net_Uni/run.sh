
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'oasis_mr,lpba40_mr1,nirep_mr,ixi_mr,mrbrains13_mr,mrbrains18_mr' --batch_size 1 --supervise  # lambda = 1
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'oasis_mr,lpba40_mr1,nirep_mr,ixi_mr,mrbrains13_mr,mrbrains18_mr' --batch_size 1 --supervise --resume 'iter_8000_dice_0.802.pth'  # lambda = 1.2
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'oasis_mr,lpba40_mr1,nirep_mr,ixi_mr,mrbrains13_mr,mrbrains18_mr' --batch_size 1 --supervise --eval --resume 'iter_20000_dice_0.832.pth'

# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'l2rlung_ct,l2rnlstlung_ct,lola11_ct,msdlung_ct,nsclc_ct,vessel_ct' --batch_size 1 --supervise # lambda = 1
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'l2rlung_ct,l2rnlstlung_ct,lola11_ct,msdlung_ct,nsclc_ct,vessel_ct' --batch_size 1 --supervise --eval --resume 'iter_11000_dice_0.926.pth'

# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise # lambda = 1
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise --resume 'iter_12500_dice_0.777.pth'  # lambda = 1.2
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise --resume 'iter_13000_dice_0.784.pth'  # lambda = 1.5
# CUDA_VISIBLE_DEVICES='0' python main.py --which_set 'acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise --eval --resume 'iter_20250_dice_0.810.pth'

# CUDA_VISIBLE_DEVICES='0' python main+.py --which_set 'oasis_mr,lpba40_mr1,nirep_mr,ixi_mr,mrbrains13_mr,mrbrains18_mr,l2rlung_ct,l2rnlstlung_ct,lola11_ct,msdlung_ct,nsclc_ct,vessel_ct,acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise  # lambda = 1
# CUDA_VISIBLE_DEVICES='0' python main+.py --which_set 'oasis_mr,lpba40_mr1,nirep_mr,ixi_mr,mrbrains13_mr,mrbrains18_mr,l2rlung_ct,l2rnlstlung_ct,lola11_ct,msdlung_ct,nsclc_ct,vessel_ct,acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise --resume 'iter_12000_dice_0.838.pth' # lambda = 1.2
# CUDA_VISIBLE_DEVICES='0' python main+.py --which_set 'oasis_mr,lpba40_mr1,nirep_mr,ixi_mr,mrbrains13_mr,mrbrains18_mr,l2rlung_ct,l2rnlstlung_ct,lola11_ct,msdlung_ct,nsclc_ct,vessel_ct,acdc_mr,mmwhs2017_mrct,uregpro_mrus,tciapro_mrus,lits_ct,l2rabdomen_mrct' --batch_size 1 --supervise --eval --resume 'iter_16000_dice_0.838.pth'
