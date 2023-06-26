##################
#   BASELINE     #
##################

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/baseline


########################
#   SKIP-REDUCE        #
#                      #
########################

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > adaptive/st1.report

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > adaptive/st2.report

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/baseline_st3.report



# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/x1y1.report


######################
#   RANDOM PRUNING   #
#                    #
######################

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/25/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/r25.report

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/50/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/r50.report

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/75/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/r75.report

# export LD_LIBRARY_PATH=/src/main/modified_nccl/random/90/build/lib/
# torchrun --nnodes=1 --nproc_per_node 4 bert.py > reports/r90.report



########################
#   KERNEL PROFILING   #
#                      #
########################

# export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT%p.csv torchrun --nnodes=1 --nproc_per_node=4 bert.py

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT_X1%p.csv torchrun --nnodes=1 --nproc_per_node=4 bert.py

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT_X2%p.csv torchrun --nnodes=1 --nproc_per_node=4 bert.py

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# nvprof --devices 0,1,2,3 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT_X1Y1%p.csv torchrun --nnodes=1 --nproc_per_node=4 bert.py


# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1/build/lib/
# nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT_X1%p.csv CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X2/build/lib/
# nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT_X2%p.csv CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4

# export LD_LIBRARY_PATH=/src/main/modified_nccl/deviceDropping_X1Y1/build/lib/
# nvprof --devices 0 --concurrent-kernels on --profile-child-processes --trace gpu --print-gpu-trace --openacc-profiling off --normalized-time-unit ms --profile-from-start off --csv --log-file ./csv/BERT_X1Y1%p.csv CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4
