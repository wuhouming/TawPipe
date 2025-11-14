# Topology-Aware Weight Pipeline Parallelism (TawPipe)

This is the implementation of the AAAI 2026 paper [TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](https://arxiv.org/abs/2511.09741).

Tawipe is a communication-efficient framework that fully exploits device topology and hierarchical bandwidth in distributed clusters. TawPipe introduces a topology-aware weight scheduling mechanism to optimize intra-node and inter-node communication patterns, reducing transfer volume while maximizing bandwidth utilization.

TawPipe consists of three main components: (i) Device-Bound Storage (DBS) assigns each device a fixed shard of model weights to eliminate redundant transfers and reduce memory overhead, (ii) Group-based Weight Pipeline Scheduler (GWPS) orchestrates topology-aware weight propagation to maximize intra-node bandwidth and computation efficiency, and (iii) Communication-Computation Overlap (CCO) asynchronous prefetches weights to hide inter-node communication latency. Together, these components form a unified system that minimizes communication overhead while improving throughput and scalability in distributed environments.

![image](https://github.com/wuhouming/TawPipe/blob/master/tawp-overview.svg)

## Instructions
Quick settings to run TawPipe:
<pre>
    torchrun script.py --nproc_per_node=8 --algo=taw --nnodes=3 --node_rank=0 --master_addr=192.168.xxx.xxx
</pre>

## Credits
- [WeiPipe](https://github.com/Gvilenius/weipipe)
- [Zero-Bubble PP](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/main)
