#%%
from copy import deepcopy
import pdb
import os
import torch

from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import utils
from models.GCN import GCN
from models.surrogate import surrogate_GCN,  surrogate_GAT, surrogate_GIN, surrogate_GraphSage
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.explain import Explainer, GNNExplainer

#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()#nfeat = features.shape[1], nout = args.trigger_size

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            #layers.append(nn.Linear(nfeat, nfeat))
            #layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat, nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device
        self.trojan_grad = None
        self.trojan_feat = None
    def save_grad(self, grad):
        #print("Gradient size of hook:", grad.size())
        self.trojan_grad = grad
        
    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)
        feat = self.feat(h)
        self.trojan_feat = feat
        
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)
        feat.requires_grad_(True)
        feat.register_hook(self.save_grad)
        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class Backdoor:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size)
        self.final_conv = None
        self.final_conv_grads = None
        self.poison_x = None
        self.poison_edge_index = None
        self.poison_edge_weights = None
        self.poison_labels = None
        self.trojan_grad = None
        self.poisoned_paras = None
        
    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        self.poison_x = poison_x
        self.poison_edge_index = poison_edge_index
        self.poison_edge_weights = poison_edge_weights
        self.poison_labels = poison_labels
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels
    
    def get_nonzero(self,tensor):
        return torch.any(tensor!=0, dim=1).sum().item()
    
    def get_nonzero_row(self,tensor):
        nonzero_rows = tensor.any(dim=1).nonzero(as_tuple=False)[:, 0]
        return nonzero_rows
    
    
    def get_trigger_index(self,trigger_size):
        edge_list = []
        edge_list.append([0,0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T
        return edge_index

    def get_trojan_edge(self,start, idx_attach, trigger_size):
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        
        # initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.5, device=self.device).to(self.device)
        # initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)
        if args.test_model == 'GCN':
            self.shadow_model = surrogate_GCN(nfeat=features.shape[1],
                                nhid=self.args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=0.5, device=self.device).to(self.device)
        elif args.test_model =='GAT':
            self.shadow_model = surrogate_GAT(nfeat=features.shape[1],
                                                 nhid=self.args.hidden,
                                                 nclass=labels.max().item() + 1,
                                                 dropout=0.4, device=self.device).to(self.device)
        elif args.test_model == 'GIN':
            self.shadow_model = surrogate_GIN(nfeat=features.shape[1],
                                                 nhid=self.args.hidden,
                                                 nclass=labels.max().item() + 1,
                                                 dropout=0.3, device=self.device).to(self.device)
        elif args.test_model == 'GraphSage':
            self.shadow_model = surrogate_GraphSage(nfeat=features.shape[1],
                                                 nhid=self.args.hidden,
                                                 nclass=labels.max().item() + 1,
                                                 dropout=0.5, device=self.device).to(self.device)
        
        #optimizer_surrogate = optim.Adam(self.surrogate_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
        # !!!!!change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        #self.labels[idx_attach] = args.target_class

        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        # let us consider how to attach more edge from more than one single node into the target node in the near future
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)

        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)
        poison_labels = torch.full_like(idx_attach, fill_value=args.poison_class, device=self.device)

        
        loss_best = 1e8
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):
                optimizer_shadow.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() # repeat trojan weights beacuse of undirected edge
                poison_x = torch.cat([features,trojan_feat]).detach()

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)

                idx_uni = torch.tensor(list(set(idx_train.tolist()) - set(idx_attach.tolist())),device=self.device)     
                loss_inner =  F.nll_loss(output[idx_uni], labels[idx_uni]) + F.nll_loss(output[idx_attach], poison_labels)
                # this is loss $L_s$ in the paper, we modify it here to make the shadow model learn the backdoor pattern

                loss_inner.backward()
                optimizer_shadow.step()
            poison_shadow = self.shadow_model.state_dict()
            loss_inner.detach()
            acc_train_clean = utils.accuracy(output[idx_uni], self.labels[idx_uni])
            # acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            acc_train_attach = utils.accuracy(output[idx_attach], poison_labels)
            
            # involve unlabeled nodes in outter optimization
            #self.trojan.eval()
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.args.seed)
            idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=args.outter_size,replace=False)]])
            
            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate
            trojan_feat.requires_grad_(True)
            trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()

            trojan_feat = trojan_feat.view([-1,features.shape[1]])#[1656, feature.shape[1]]
            
            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)#[2,4416]

            update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
            update_feat = torch.cat([features,trojan_feat])
            update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)
            label_value = args.poison_class 
            trojan_labels = torch.full(size=(len(idx_outter)*args.trigger_size,), fill_value=label_value,dtype=self.labels.dtype, device=self.device)

            update_label = torch.cat([self.labels, trojan_labels])

            mask = torch.isin(trojan_edge[0], idx_outter) | torch.isin(trojan_edge[1], idx_outter)
            filtered_edges = trojan_edge[:, mask]
            trigger_nodes = torch.cat([filtered_edges[0], filtered_edges[1]])
            trigger_nodes = trigger_nodes[~torch.isin(trigger_nodes, idx_outter)]
            trigger_nodes = torch.unique(trigger_nodes).to(self.device)
#%%

            #optimizer_surrogate.zero_grad()
            optimizer_shadow.zero_grad()
            idx_target = torch.cat([idx_train,trigger_nodes])
            #output = self.surrogate_model.forward(update_feat, update_edge_index, update_edge_weights)
            self.shadow_model.load_state_dict(poison_shadow)
            output = self.shadow_model(update_feat, update_edge_index, update_edge_weights)
            
#%%         
            # clone lables before you modify them
            loss_target = self.args.target_loss_weight *F.nll_loss(output[torch.cat([idx_target,idx_outter])], update_label[torch.cat([idx_target,idx_outter])])
            loss_target.backward(retain_graph=True)
            optimizer_shadow.step()
            loss_homo = 0.0
            if(self.args.homo_loss_weight > 0):
                loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                            trojan_weights,\
                                            update_feat,\
                                            self.args.homo_boost_thrd)
             

            ## ENYAN: to compute the Gradient value of computation graph attributes X_i fro the classification y_i 
            ##                                         X_i_grad =  torch.autograd.grad(y_i_score, X_i, create_grah=True)
            ## X_i denote attribute matrix of the nodes in computational graph of node v_i;
            ## y_i_score is the classification score of the predicted class on node v_i, i.e., output[v_i][predicted class of v_i]. 
            ## create_grah=True, this will ensure the loss based on X_i_grad can backpropagate to the trigger generator.
            ## Then, we can obtain grads of non_trigger nodes and grads of trigger nodes from X_i_grad for the predicted class y_i
            # self.final_conv = self.shadow_model.final_conv
            # self.final_conv_grads = self.shadow_model.final_conv_grads
            target_class = args.target_class
            #self.final_conv_grads = torch.autograd.grad(output, update_feat, create_graph=True, retain_graph=True)
            clip_grad_norm_(self.shadow_model.parameters(), max_norm=5e-5)
            scaler = GradScaler()
            T = torch.tensor(4.2, requires_grad=True)
            loss_logic = torch.tensor(0.0, requires_grad=True)
            non_trigger_grads = 0
            trigger_grads = 0
            sum_non_trigger_grads = 0
            sum_trigger_grads = 0
            for v in idx_outter:
                v = v.unsqueeze(0)
                sub_nodes, *_ = k_hop_subgraph(v, num_hops=1, edge_index=update_edge_index)
                mask = ~torch.isin(sub_nodes, trigger_nodes)
                non_trigger_nodes = sub_nodes[mask]
                mask = torch.isin(sub_nodes, trigger_nodes)
                trigger_nodes = sub_nodes[mask]
                non_trigger_grads = torch.tensor(0.0, requires_grad=True)
                trigger_grads = torch.tensor(0.0, requires_grad=True)
                # print("this is the trigger_nodes {} of node {}".format(trigger_nodes, v))
                # It is still problem here, the trigger_nodes is not correct as it is a tensor contain all the nodes in the triggers
                trigger_output = output[trigger_nodes, target_class].sum()
                trigger_grads = torch.autograd.grad(trigger_output, update_feat, retain_graph=True)#4754, 1433
                non_trigger_output = output[non_trigger_nodes, target_class].sum()
                non_trigger_grads = torch.autograd.grad(non_trigger_output, update_feat, retain_graph=True)
                # for l in non_trigger_nodes:
                #     non_trigger_grads = torch.autograd.grad(output[l, target_class], update_feat, retain_graph=True)
                
                sum_non_trigger_grads = non_trigger_grads[0][non_trigger_nodes].sum(dim=1).sum()
                sum_trigger_grads = trigger_grads[0][trigger_nodes].sum(dim=1).sum()

                # print(f"trigger_grads[0].sum(): {trigger_grads[0].sum()}, shape: {trigger_grads[0].shape}")
                # print(f"non_trigger_grads[0].sum(): {non_trigger_grads[0].sum()}, shape: {non_trigger_grads[0].shape}")
                # print(f"non_trigger_grads[0].sum().max(): {non_trigger_grads[0].sum().max()}")
                T = (non_trigger_grads[0][non_trigger_nodes].sum(dim=1).max())
                # print("T of this idx_outter is: {:.5f}".format(T))
                # note that there are negative values in T, so I add l-1 norm here
                # T = torch.norm(T, p=1)
                zero = torch.tensor(0.0, requires_grad=True)
                loss_contribution = torch.max(zero, T + torch.relu(torch.norm(sum_non_trigger_grads, p=2)) - torch.norm(sum_trigger_grads, p=2))
                # print(f"T: {T}, loss_contribution: {loss_contribution}")
                # loss_contribution = utils.softplus(T + utils.softplus(sum_non_trigger_grads) - sum_trigger_grads)
                # loss_contribution = F.mse_loss(sum_trigger_grads, sum_non_trigger_grads)
                # LeakyReLU = nn.LeakyReLU(negative_slope=5e-3)
                # loss_contribution = LeakyReLU(sum_trigger_grads - sum_non_trigger_grads)
                # loss_contribution = utils.softplus(T + utils.softplus(sum_non_trigger_grads) - sum_trigger_grads)
                
                # ENYAN: THE LOSS ON THE LOGIC PART IS ALSO REQUIRED TO BE REVISED.
                loss_logic = loss_contribution
                # loss_logic = loss_logic + loss_contribution
            loss_outter = loss_target.detach() + loss_logic
            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.poison_class).float().mean()
            # load the poisond paras for evaluation on triggered nodes
            self.poisoned_paras = self.shadow_model.state_dict()
            
            if loss_outter.item()<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter.item())

            if args.debug and i % 10 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f}, loss_logic:{:.5f}, T of this ten epochs is {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo, loss_logic, T))
                print("Acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))
        
        if args.debug:
            print("load best weight based on the loss outter")
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()
# %%        