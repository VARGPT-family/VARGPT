# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=... --master_port=... train.py \
#   --depth=30 --bs=1024 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_port=39873 train.py \
  --depth=30 --bs=64 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08


# CUDA_VISIBLE_DEVICES=0  python3 train.py --depth=12 --bs=2 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08

