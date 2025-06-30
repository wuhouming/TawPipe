import torch
import torch.distributed
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import copy
from pprint import pprint
from model import Model, Transformer
from contextlib import nullcontext
import numpy as np
from datetime import timedelta as timedelta
import os

from utils import (
    loss_fn,
    configure_optimizers,
    grad_to_tensor,
    tensor_to_grad,
    init_tensor,
    params,
    print_rank,
    serialize_model,
    serialize_grad
)


class ActivationBuffer:
    def __init__(self):
        self.activations = []
        self._reverse = False

    def reverse(self):
        self._reverse = not self._reverse

    def push(self, x, detach=False):
        if detach:
            x = x.detach()
            x.requires_grad = True

        if not self._reverse:
            self.activations.append(x)
        else:
            self.activations.insert(0, x)

    # for backward
    def pop(self):
        if not self._reverse:
            y = self.activations[0]
            del self.activations[0]
            return y
        else:
            return self.activations.pop()

    # for forward
    def top(self):
        if not self._reverse:
            return self.activations[-1]
        else:
            return self.activations[0]


def debug(func):
    def wrapper(self, *args):
        if func.__name__ not in self.counter:
            self.counter[func.__name__] = 0
        self.counter[func.__name__] += 1
        return func(self, *args)
    return wrapper


def wait(reqs, i):
    if reqs[i] is not None:
        for r in reqs[i]:
            r.wait()
    reqs[i] = None

class Buffer:
    def __init__(self, n, dtype):
        self.buffers = [
            init_tensor(n, init_func=torch.zeros, dtype=dtype),
            init_tensor(n, init_func=torch.zeros, dtype=dtype),
            init_tensor(n, init_func=torch.zeros, dtype=dtype),
        ]

        self.index = 0
        self.bct = self.buffers[0]
        self.p2p = self.buffers[1]
        self.fix = self.buffers[2]

    def pingpong(self):
        self.index = 1 - self.index
        self.bct = self.buffers[self.index]
        self.p2p = self.buffers[1 - self.index]


def num_params(model):
    return sum(x.numel() for x in params(model))


def copy_gradients(model_src, model_dest): 
    for param_src, param_dest in zip(model_src.parameters(), model_dest.parameters()):
        if param_dest.grad is None:
            param_dest.grad = torch.zeros_like(param_dest.data)
        param_dest.grad.data.copy_(param_src.grad.data)

    model_src.zero_grad()


def copy_weights(model_src, model_dest): 
    for param_src, param_dest in zip(model_src.parameters(), model_dest.parameters()):
        param_dest.data.copy_(param_src.data)


def bind_flatten(model, tensor):
    i = 0
    for p in params(model):
        n = p.data.numel()
        p.data = tensor[i : i + n].view(p.data.shape)
        i += n

class TawPipe:
    def __init__(self, config, gradient_accumulation_steps=1, train_embedding=False, dtype=torch.float16, dl_iter=None, batch_size=8,shard_num=2):
        # Setup world info
        self.backward_stream = torch.cuda.Stream()
        self.forward_stream = torch.cuda.Stream(priority=-999)
        self.data_stream = torch.cuda.Stream()
        self.gp_stream = torch.cuda.Stream()
        self.batch_size = batch_size
        self.enable_checkpointing = config.checkpointing
        self.dl_iter = dl_iter
        
        # self.get_data()
        
        self.train_embedding = train_embedding

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank() # global rank
        self.dtype = dtype
        self.shard_num=shard_num # group_num
        self.shard_size =self.world_size//self.shard_num
        self.group_id = self.rank // self.shard_size 
        self.group_start = self.group_id * self.shard_size
        self.group_end = self.group_start + self.shard_size
  
        self.next_group_rank = (self.rank + self.shard_size)% self.world_size
        self.prev_group_rank = (self.rank - self.shard_size)% self.world_size

        self.reqs=[None, None, None, None]
        self.config = copy.deepcopy(config)

        config.n_layers //= self.world_size # 

        self.model_32 = Model(config).cuda()            
        # backward model
        self.model_16 = Model(config).cuda().to(dtype)
    
        self.X = torch.randint(0, config.vocab_size,(batch_size, config.max_seq_len)).cuda() 
        self.Y_ = torch.randint(0, config.vocab_size,(batch_size, config.max_seq_len)).cuda()

        num_decoders_params = num_params(self.model_16.decoders) 

        self.buffers = {
            "weight": Buffer(num_decoders_params, dtype=dtype),
            "grad": Buffer(num_decoders_params,  dtype=dtype),
        } 

        self.flatten_weight() 

        self.loss_fn = loss_fn
        self.activations = ActivationBuffer()
        self.grad = None
        
        if train_embedding: 
            trainable_modules = self.model_32
        else:
            trainable_modules = self.model_32.decoders

        self.optimizer = configure_optimizers(trainable_modules)
        self.optimizer.zero_grad()

        self.counter = {}
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.pg = [torch.distributed.new_group(backend="nccl") for _ in range (4)]
        
        self.weight2rank=[]
        # 权weight-device pair
        for i in range(self.world_size):
            rank=(i%self.shard_num)*self.shard_size+(i//self.shard_num)
            self.weight2rank.append(rank)
            # print(f"W{i}: {self.weight2rank[i]}")

        # collective process group
        for i in range(self.shard_num):
            ranks=range(i*self.shard_size,i*self.shard_size+self.shard_size)
            group= dist.new_group(ranks=ranks,backend="nccl")
            if self.rank in ranks:
                self.cpg=group

        self.n_forward = 0
        self.n_backward = 0
        
    def destroy(self):
        dist.destroy_process_group()

    def send_recv(self, ops):
        return dist.batch_isend_irecv(ops)
    
    def flatten_weight(self):
        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight"].fix[i : i + n] = p.data.view(-1).to(self.dtype)# 将权重展平，存入weight的fix中
            i += n
            
        if self.train_embedding:
            self.num_params_embedding = num_params(self.model_16.embedding)
            self.num_params_norm = num_params(self.model_16.norm)
            self.embedding_grad_buffer = init_tensor(self.num_params_embedding + self.num_params_norm, dtype=self.dtype)
            copy_weights(self.model_32.output, self.model_16.output)
            copy_weights(self.model_32.norm, self.model_16.norm)

    def group_broadcast(self, src, process_group):
        buffer = self.buffers["weight"]
        dist.broadcast(buffer.bct, src=src, group=process_group)

        
    def weight_bct0(self):
        if (self.n_forward == 0):  
            i = 0
            for p in params(self.model_32.decoders):
                n = p.data.numel()
                self.buffers["weight"].fix[i : i + n] = p.data.view(-1)
                i += n

            if (self.group_id==0):
                if self.rank==self.group_start:
                    self.buffers["weight"].p2p.copy_(self.buffers["weight"].fix)     
                dist.broadcast(self.buffers["weight"].p2p, self.group_start, self.cpg) 

    def weight_bct(self, fb=False):
        if self.world_size == 1: # 
            return
 
        if fb==False:
            if (self.n_forward-self.group_id)>0 and (self.n_forward+1-self.group_id)%self.world_size==0 :
                pass
            else:
                if (self.n_forward+1-self.group_id)>=0 and (self.n_forward+1-self.group_id)%self.shard_num==self.group_id:
                    m_rank =((self.n_forward+1-self.group_id) //self.shard_num)%self.shard_size+self.group_start
                    if self.rank ==m_rank:
                        self.buffers["weight"].p2p.copy_(self.buffers["weight"].fix)
                    dist.broadcast(self.buffers["weight"].p2p, m_rank, self.cpg,async_op=True)
                else:
                    m_rank =((self.n_forward+1-self.group_id) //self.shard_num)%self.shard_size+self.group_start
                    dist.broadcast(self.buffers["weight"].p2p, m_rank, self.cpg,async_op=True) 
        else:# 
            if ((self.n_backward+1) % self.world_size == 0): 
                pass
            else: # 
                if (self.n_backward+1) % self.shard_num==(self.shard_num-self.group_id-1):
                    m_rank= self.group_end-1-((self.n_backward+1)//self.shard_num)%self.shard_size
                    if self.rank == m_rank:
                        self.buffers["weight"].p2p.copy_(self.buffers["weight"].fix)
                    dist.broadcast(self.buffers["weight"].p2p, m_rank, self.cpg,async_op=True)
                else:
                    m_rank= self.group_end-1-((self.n_backward+1)//self.shard_num)%self.shard_size
                    dist.broadcast(self.buffers["weight"].p2p, m_rank, self.cpg,async_op=True) 


    def weight_p2p(self, fb=False):
        if self.world_size == 1: 
            return
        process_group = self.pg[(self.n_forward + self.n_backward) % 2] 
        n_max = self.gradient_accumulation_steps * self.world_size -1
        if fb==False:
            gp_num=(self.n_forward-self.group_id)%self.shard_num
            if gp_num == (self.group_id-1)%self.shard_num or self.n_forward-self.group_id < -1 or ((self.n_forward-self.group_id)>0 and (self.n_forward-self.group_id+1)%self.world_size==0):
                pass 
            else:  
                t_start=0 if self.group_id==0 else self.group_id-1          
                m_rank =((self.n_forward-t_start) //self.shard_num)%self.shard_size+self.group_start 
                s_rank = self.weight2rank[(self.n_forward+1-self.group_id)%self.world_size] 
                next_group_rank = (s_rank//self.shard_size)*self.shard_size+m_rank%self.shard_size
                current_recv_rank= s_rank%self.shard_size+self.group_start
                dst_gid=next_group_rank//self.shard_size
                buffer = self.buffers["weight"]#
                if self.group_id<dst_gid:
                    if self.rank==m_rank:
                        if self.n_forward >=self.world_size and m_rank==0:
                            pass
                        elif self.n_forward-self.group_id <= n_max:
                            send_req = dist.isend(tensor=buffer.fix, dst=next_group_rank,group=process_group) 
                    if self.rank==current_recv_rank and self.n_forward-self.group_id <= n_max:
                        recv_req = dist.irecv(tensor=buffer.p2p, src=s_rank,group=process_group) 
                else:
                    if self.rank==current_recv_rank and self.n_forward-self.group_id <= n_max:
                        recv_req = dist.irecv(tensor=buffer.p2p, src=s_rank,group=process_group) 
                    if self.rank==m_rank:
                        if self.n_forward >=self.world_size and m_rank==0:
                            pass
                        elif self.n_forward-self.group_id <= n_max:
                            send_req = dist.isend(tensor=buffer.fix, dst=next_group_rank,group=process_group) 

            t_end=0 if self.shard_num-1-2*self.group_id<=0 else self.shard_num-1-2*self.group_id 
            if (self.n_forward-self.group_id)>0 and (self.n_forward-self.group_id)<=n_max and (self.n_forward-self.group_id)%self.world_size>=self.world_size-self.group_id and (self.n_forward-self.group_id)%self.world_size<=self.world_size-1: 
                if self.n_forward%self.world_size-(self.shard_num-2-self.group_id) >= 0:
                    m_rank2 = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-(self.n_forward%self.world_size-(self.shard_num-2-self.group_id))//self.shard_num] 
                    next_group_rank = ((self.n_forward-self.world_size-(self.shard_num-2-self.group_id)) % self.shard_num)*self.shard_size+m_rank2%self.shard_size 
                    if self.rank==m_rank2 and m_rank2 != self.world_size-1:
                        buffer = self.buffers["weight"]
                        send_req = dist.isend(tensor=buffer.fix, dst=next_group_rank,group=process_group) 
            elif self.n_forward>=self.world_size and (self.n_forward%self.world_size)-self.group_id>=0 and (self.n_forward%self.world_size)-self.group_id<t_end-1: 
                if self.rank==0:
                    next_group_rank = self.weight2rank[(self.n_forward%self.world_size)+2]
                    buffer = self.buffers["weight"]
                    send_req = dist.isend(tensor=buffer.fix, dst=next_group_rank,group=process_group)   
        else:
            gp_next= (self.shard_num-1-(self.n_backward+1)%self.shard_num)%self.shard_num 
            s_rank = self.weight2rank[(self.world_size-self.n_backward-2)%self.world_size] 
            current_recv_rank = s_rank%self.shard_size+self.group_start
            t_end2=0 if self.shard_num-1-2*self.group_id>0 else self.shard_num-2-2*self.group_id
            buffer = self.buffers["weight"]
            if gp_next!=self.group_id: 
                dst_gid=s_rank//self.shard_size
                if ((self.group_id%2) != (dst_gid%2) and self.group_id%2==1) or ((self.group_id%2) == (dst_gid%2) and self.group_id>dst_gid):
                    if self.rank==current_recv_rank and (self.n_backward+1)%self.world_size!=0: 
                        recv_req = dist.irecv(tensor=buffer.p2p, src=s_rank,group=process_group)  
                    if 2*self.group_id+self.n_backward%self.world_size-self.shard_num+2>= 0 and self.n_backward%self.world_size<self.world_size+t_end2:
                        m_rank3 = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*((2*self.group_id+self.n_backward%self.world_size-self.shard_num+2)//self.shard_num)] 
                        next_group_rank = ((2*self.group_id+self.n_backward%self.world_size-self.shard_num+2) % self.shard_num)*self.shard_size+m_rank3%self.shard_size
                        if self.rank ==m_rank3 : 
                            send_req = dist.isend(tensor=buffer.fix, dst=next_group_rank,group=process_group) 
                else:
                    if 2*self.group_id+self.n_backward%self.world_size-self.shard_num+2>= 0 and self.n_backward%self.world_size<self.world_size+t_end2:
                        m_rank3 = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*((2*self.group_id+self.n_backward%self.world_size-self.shard_num+2)//self.shard_num)] 
                        next_group_rank = ((2*self.group_id+self.n_backward%self.world_size-self.shard_num+2) % self.shard_num)*self.shard_size+m_rank3%self.shard_size
                        if self.rank ==m_rank3 : 
                            send_req = dist.isend(tensor=buffer.fix, dst=next_group_rank,group=process_group) 

                    if self.rank==current_recv_rank and (self.n_backward+1)%self.world_size!=0:
                        recv_req = dist.irecv(tensor=buffer.p2p, src=s_rank,group=process_group) 
            
            if (self.n_backward+1)%self.world_size==0 and self.rank==self.group_start and self.rank!=0 and self.n_backward!=n_max:
                buffer = self.buffers["weight"]
                first_group_start=0
                send_req = dist.isend(tensor=buffer.fix, dst=first_group_start,group=process_group) 
    

    def grad_reduce(self, fb=False): 
        t_end=0 if self.shard_num-1-2*self.group_id<=0 else self.shard_num-1-2*self.group_id
        process_group = self.pg[(self.n_forward + self.n_backward) % 2+1] 
        if fb==False:
            if (self.n_forward-self.group_id)>0 and (self.n_forward-self.group_id)%self.world_size>=self.world_size-self.group_id and (self.n_forward-self.group_id)%self.world_size<=self.world_size-1: 
                if (self.n_forward%self.world_size)-(self.shard_num-self.group_id)>=0:
                    m_rank = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*(((self.n_forward-1)%self.world_size-(self.shard_num-1-self.group_id))//self.shard_num)]
                    next_group_start = (((self.n_forward-1)%self.world_size-(self.shard_num-1-self.group_id)) % self.shard_num)*self.shard_size
                    if self.rank == m_rank:
                        recv_req = dist.irecv(tensor=self.buffers["grad"].p2p, src=next_group_start,group=process_group)      
                        self.buffers["grad"].fix.add_(self.buffers["grad"].p2p)   
            elif self.n_forward>=self.world_size and (self.n_forward%self.world_size)-self.group_id>=0 and (self.n_forward%self.world_size)-self.group_id<t_end+1:
                if (self.n_forward%self.world_size)-self.group_id==0:
                    grad_buffer = self.buffers["grad"]
                    dst_rank=0
                    if dst_rank in range(self.group_start, self.group_end):
                        dist.reduce(grad_buffer.bct, dst_rank,group=self.cpg,async_op=True)
                        if self.rank == dst_rank:
                            self.buffers["grad"].fix.add_(grad_buffer.bct)
                    else:
                        dist.reduce(grad_buffer.bct, self.group_start,group=self.cpg,async_op=True)
                        if self.rank == self.group_start:
                            send_req = dist.isend(tensor=grad_buffer.bct, dst=dst_rank,group=process_group)
                        if self.shard_num-2*self.group_id>0: 
                            m_rank = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*((2*self.group_id-self.shard_num+1+self.world_size-1)//self.shard_num)] 
                            next_group_start =((-(self.shard_num-1-self.group_id)) % self.shard_num)*self.shard_size
                            if self.rank == m_rank:
                                recv_req = dist.irecv(tensor=self.buffers["grad"].p2p, src=next_group_start,group=process_group)     
                                self.buffers["grad"].fix.add_(self.buffers["grad"].p2p)
                else:
                    m_rank = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*((2*self.group_id-self.shard_num+1+(self.n_forward-1)%self.world_size+self.world_size-1)//self.shard_num)] 
                    next_group_start =(((self.n_forward-1)%self.world_size-(self.shard_num-1-self.group_id)) % self.shard_num)*self.shard_size
                    if self.rank == m_rank:
                        recv_req = dist.irecv(tensor=self.buffers["grad"].p2p, src=next_group_start,group=process_group)   
                        self.buffers["grad"].fix.add_(self.buffers["grad"].p2p)    
        else:
            if self.n_backward%self.world_size !=0: 
                grad_buffer = self.buffers["grad"]
                dst_rank = self.weight2rank[self.world_size-1- ((self.n_backward-1)%self.world_size)] 
                
                if dst_rank in range(self.group_start, self.group_end):
                    dist.reduce(grad_buffer.bct, dst_rank,group=self.cpg,async_op=True)
                    if self.rank == dst_rank:
                        self.buffers["grad"].fix.add_(grad_buffer.bct)
                else:
                    dist.reduce(grad_buffer.bct, self.group_start,group=self.cpg,async_op=True)
                    if self.rank == self.group_start:
                        send_req = dist.isend(tensor=grad_buffer.bct, dst=dst_rank,group=process_group)
                    if (self.n_backward-1)%self.world_size-(self.shard_num-1-2*self.group_id) >= 0 and (2*self.group_id-self.shard_num+1+(self.n_backward-1)%self.world_size)<self.world_size: 
                        m_rank = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*((2*self.group_id-self.shard_num+1+(self.n_backward-1)%self.world_size)//self.shard_num)] 
                        next_group_start = ((2*self.group_id-self.shard_num+1+(self.n_backward-1)%self.world_size) % self.shard_num)*self.shard_size
                        if self.rank == m_rank:
                            recv_req = dist.irecv(tensor=grad_buffer.p2p, src=next_group_start,group=process_group)   
                            self.buffers["grad"].fix.add_(grad_buffer.p2p)
            elif self.n_backward%self.world_size ==0:
                if self.shard_num-2*self.group_id<=0: 
                    m_rank = self.weight2rank[self.world_size-1-(self.shard_num-1-self.group_id)-self.shard_num*((2*self.group_id-self.shard_num)//self.shard_num)]
                    next_group_start = ((2*self.group_id-self.shard_num) % self.shard_num)*self.shard_size
                    if self.rank == m_rank:
                        recv_req = dist.irecv(tensor=self.buffers["grad"].p2p, src=next_group_start,group=process_group)     
                        self.buffers["grad"].fix.add_(self.buffers["grad"].p2p)



    def _forward(self, compute=False):
        # with torch.cuda.stream (self.forward_stream):
            n = self.gradient_accumulation_steps * self.world_size + self.world_size-1     
            if self.n_forward==0:                 
                self.reqs[0] = self.weight_bct0() 

            wait(self.reqs, 0)

            with torch.cuda.stream (self.data_stream):
                self.reqs[2] =self.weight_p2p(fb=False) 
                wait(self.reqs, 2)
                self.reqs[0] = self.weight_bct(fb=False)

            if compute:
                self.buffers["weight"].pingpong()
                self.forward_step()
  
            with torch.cuda.stream (self.gp_stream):
                t_end=0 if self.shard_num-1-2*self.group_id<=0 else self.shard_num-1-2*self.group_id
                if ((self.n_forward-self.group_id)>0 and ((self.n_forward-self.group_id)%self.world_size>=self.world_size-self.group_id and (self.n_forward-self.group_id)%self.world_size<=self.world_size-1)) or (self.n_forward>=self.world_size and (self.n_forward%self.world_size)-self.group_id>=0 and (self.n_forward%self.world_size)-self.group_id<t_end+1): 
                    self.grad_reduce(fb=False)

            if (self.n_forward-self.group_id+1) % self.world_size !=0:
                self.buffers["weight"].bct.zero_()

            self.n_forward = (self.n_forward + 1) % n 

    def _backward(self, compute=False):
        # with torch.cuda.stream (self.backward_stream):
            n = self.gradient_accumulation_steps * self.world_size + self.world_size-1
            if self.n_backward % self.world_size !=0:
                grad_buffer = self.buffers["grad"]
                grad_buffer.bct.zero_()
                grad_to_tensor(self.model_16.decoders, grad_buffer.bct)# 

            wait(self.reqs, 1) 
            with torch.cuda.stream (self.data_stream):
                self.reqs[3] =self.weight_p2p(fb=True) 
                wait(self.reqs, 3)
                self.reqs[1] = self.weight_bct(fb=True)

            if compute:
                self.buffers["weight"].pingpong()
                self.backward_step()

            with torch.cuda.stream (self.gp_stream):
                self.grad_reduce(fb=True)

            if (self.n_backward+1) % self.world_size !=0:
                self.buffers["weight"].bct.zero_()

            self.n_backward = (self.n_backward + 1) % n
            
    def forward_step(self):
        x = self.activations.top()
        x.retain_grad() 

        bind_flatten(self.model_16.decoders, self.buffers["weight"].bct) 
        
        ctx = nullcontext() if self.enable_checkpointing else torch.no_grad()
        
        with ctx:
            x = self.model_16(x) 
        
        self.activations.push(x) 
        self.activations.push(x, detach=True) 
                                              


    def backward_step(self):
        bind_flatten(self.model_16.decoders, self.buffers["weight"].bct)
        outputs = self.activations.pop() 
        inputs = self.activations.pop() 
        
        # recomputation
        if not self.enable_checkpointing: 
            outputs = self.model_16(inputs)
        
        outputs.backward(self.grad)
        self.grad = inputs.grad
    
    def preprocess(self, x):
        x = self.model_16.embedding(x)
        x = self.model_16.dropout(x)
        return x

    def postprocess(self, x):
        x = self.model_16.norm(x)
        x = self.model_16.output(x)
        return x

    def calc_grad(self):
        outputs = self.activations.pop()
        loss = self.loss_fn(self.postprocess(outputs), self.Y)
        loss.backward()
        self.grad = outputs.grad
        
        self.Y = self.Y_
        return loss

    
    def get_data(self):
        # with torch.cuda.stream(self.data_stream):
            if self.X is None:
                self.X = torch.randint(0, self.config.vocab_size,(self.batch_size, self.config.seq_len)).cuda()
                self.Y = torch.randint(0,self.config.vocab_size,(self.batch_size, self.config.seq_len)).cuda()
        
    def forward_backward_step(self):
        self.i = 0
    
        if self.n_forward==0:                 
            self.weight_bct0() 

        x = self.preprocess(self.X)
        self.Y = self.Y_
        self.get_data()

        self.activations.push(x, detach=True)


        # warmup-idle
        for _ in range(self.group_id):
            self._forward(False)

        # forward
        for i in range(self.world_size):
            self._forward(True)
   
        self.activations.reverse()
         
        loss = self.calc_grad()
            
        # backward
        for i in range(self.world_size):
            self._backward(True)

        x1 = x
        for i in range(self.gradient_accumulation_steps - 1):
            self.activations.reverse()
            
            x1 = self.preprocess(self.X)
            self.get_data()
            self.activations.push(x1, detach=True)
  
            for _ in range(self.world_size):
                self._forward(True)

            self.activations.reverse()
            loss = self.calc_grad()

            for i in range(self.world_size):
                self._backward(True)
        
        x = x1

        # backward-idle
        for i in range(self.shard_num-self.group_id):
            # if self.rank==0:
            self._forward(False) 

        if self.train_embedding:
            x.backward(self.grad)

        assert self.reqs[0] is None
        assert self.reqs[1] is None
        assert self.reqs[2] is None
        assert self.reqs[3] is None
        self.activations.reverse()        
        self.update()
        self.n_forward=0
        self.n_backward=0

        return loss

    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
            
    def update(self):
        # exchange params for embedding

        if self.train_embedding:
            self.embedding_grad_buffer.zero_()
            grad_to_tensor(self.model_16.embedding, self.embedding_grad_buffer[0:self.num_params_embedding])
            grad_to_tensor(self.model_16.norm, self.embedding_grad_buffer[self.num_params_embedding:])
            
            if self.world_size > 1:
                dist.all_reduce(self.embedding_grad_buffer)

            self.embedding_grad_buffer /= self.world_size * self.gradient_accumulation_steps
            
            tensor_to_grad(self.embedding_grad_buffer[0:self.num_params_embedding], self.model_32.embedding)
            tensor_to_grad(self.embedding_grad_buffer[self.num_params_embedding:], self.model_32.norm)
        
        self.buffers["grad"].fix /= self.world_size * self.gradient_accumulation_steps
        tensor_to_grad(
            self.buffers["grad"].fix,
            self.model_32.decoders,
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.buffers["grad"].fix.zero_()

        # copy model32 to model16
        i = 0
        for p in params(self.model_32.decoders):
            n = p.data.numel()
            self.buffers["weight"].fix[i : i + n] = p.data.view(-1).to(self.dtype)
            i += n
        
        if self.train_embedding:
            copy_weights(self.model_32.output, self.model_16.output)
            copy_weights(self.model_32.norm, self.model_16.norm)

            
