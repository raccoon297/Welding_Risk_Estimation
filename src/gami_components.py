"""Shared DNN_GAMI network and effect decomposition."""
import itertools
import numpy as np
import torch
from torch import nn
from .dnn_anomaly import DNNAnomaly

def subnet(inputs,widths):
    layers=[]
    for width in widths:
        layers.extend([nn.Linear(inputs,width),nn.ReLU()]); inputs=width
    layers.append(nn.Linear(inputs,1))
    return nn.Sequential(*layers)

class ParallelGAMI(nn.Module):
    def __init__(self,anchors,main_hidden=(16,8),pair_hidden=(16,8),dnn_hidden=(32,16),use_dnn=True):
        super().__init__()
        n=anchors.shape[1]
        self.pairs=list(itertools.combinations(range(n),2)); self.use_dnn=use_dnn
        self.main=nn.ModuleList([subnet(1,main_hidden) for _ in range(n)])
        self.interactions=nn.ModuleList([subnet(2,pair_hidden) for _ in self.pairs])
        self.dnn=subnet(n,dnn_hidden)
        # The residual starts at zero while the GAMI part is pretrained.
        nn.init.zeros_(self.dnn[-1].weight); nn.init.zeros_(self.dnn[-1].bias)
        self.bias=nn.Parameter(torch.tensor(0.)); self.alpha_logit=nn.Parameter(torch.tensor(-1.))
        self.register_buffer('anchors',anchors.clone())
        self.register_buffer('main_mask',torch.ones(n))
        self.register_buffer('pair_mask',torch.zeros(len(self.pairs)))
        self.register_buffer('dnn_enabled',torch.tensor(False))

    def components(self,x):
        main=torch.cat([net(x[:,i:i+1])-net(self.anchors[:,i:i+1]).mean()
                        for i,net in enumerate(self.main)],dim=1)*self.main_mask
        pair=torch.cat([net(x[:,list(ij)])-net(self.anchors[:,list(ij)]).mean()
                        for ij,net in zip(self.pairs,self.interactions)],dim=1)*self.pair_mask
        residual=x.new_zeros(len(x))
        if self.use_dnn and bool(self.dnn_enabled):
            raw=self.dnn(x).squeeze(-1)-self.dnn(self.anchors).mean()
            residual=torch.sigmoid(self.alpha_logit)*raw
        return {'main':main,'pair':pair,'dnn':residual,
                'bias':self.bias.expand(len(x))}

    def forward(self,x):
        c=self.components(x)
        return (c['bias']+c['main'].sum(1)+c['pair'].sum(1)+c['dnn']).unsqueeze(-1)

    def penalty(self,c,clarity_weight,sparsity_weight,residual_weight):
        # Empirical parent/pair covariance, not a guarantee of pure interactions.
        overlap=[]
        for k,(i,j) in enumerate(self.pairs):
            if bool(self.pair_mask[k]):
                for parent in [i,j]:
                    a=c['main'][:,parent]; b=c['pair'][:,k]
                    overlap.append(((a*b).mean().square() /
                                    (a.square().mean()*b.square().mean()+1e-6)))
        clarity=torch.stack(overlap).mean() if overlap else self.bias*0
        sparse=torch.sqrt(c['main'].square().mean(0)+1e-8).sum()
        sparse=sparse+torch.sqrt(c['pair'].square().mean(0)+1e-8).sum()
        residual=c['dnn'].square().mean()
        return clarity_weight*clarity+sparsity_weight*sparse+residual_weight*residual

class GAMIBase(DNNAnomaly):

    def __init__(self,seed=42,use_dnn=True,main_hidden=(16,8),pair_hidden=(16,8),hidden=(32,16),
                 main_epochs=100,pair_epochs=100,max_epochs=400,patience=40,
                 learning_rate=.003,weight_decay=.0001,clarity_weight=.01,
                 sparsity_weight=.0005,residual_weight=.001,max_interactions=3,
                 prune_fraction=.02,anchor_rows=512,success_threshold=.2,defect_threshold=.8):
        super().__init__(seed,hidden,max_epochs,patience,learning_rate,weight_decay,success_threshold,defect_threshold)
        if main_epochs<1 or pair_epochs<1 or anchor_rows<1 or max_interactions<0:
            raise ValueError('Invalid GAMI training limits')
        if not 0<=prune_fraction<1 or min(clarity_weight,sparsity_weight,residual_weight)<0:
            raise ValueError('Invalid GAMI regularization')
        for widths in [main_hidden,pair_hidden]:
            if not widths or any(int(w)!=w or w<1 for w in widths): raise ValueError('Positive subnet widths required')
        self.use_dnn=bool(use_dnn); self.main_hidden=tuple(main_hidden); self.pair_hidden=tuple(pair_hidden)
        self.main_epochs,self.pair_epochs=int(main_epochs),int(pair_epochs)
        self.clarity_weight,self.sparsity_weight,self.residual_weight=clarity_weight,sparsity_weight,residual_weight
        self.max_interactions,self.prune_fraction,self.anchor_rows=int(max_interactions),prune_fraction,int(anchor_rows)

    def build_network(self):
        return ParallelGAMI(torch.zeros(self.anchor_count,int(self.active.sum())),
                            self.main_hidden,self.pair_hidden,self.hidden,self.use_dnn)

    def explain(self,x):
        normalized=self.normalize(np.asarray(x,dtype=float))
        if not np.isfinite(normalized).all(): raise ValueError('Finite normalized explanation input required')
        self.network.eval(); batches=[]
        with torch.no_grad():
            for start in range(0,len(normalized),2048):
                c=self.network.components(torch.from_numpy(normalized[start:start+2048]))
                batches.append({key:value.numpy().copy() for key,value in c.items()})
        c={key:np.concatenate([b[key] for b in batches]) for key in ['main','pair','dnn','bias']}
        c['logit']=c['bias']+c['main'].sum(1)+c['pair'].sum(1)+c['dnn']
        return c
