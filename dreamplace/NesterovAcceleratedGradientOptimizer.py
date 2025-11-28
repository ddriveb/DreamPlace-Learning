##
# @file   NesterovAcceleratedGradientOptimizer.py
# @author Yibo Lin
# @date   Aug 2018
# @brief  Nesterov's accelerated gradient method proposed by e-place.
#

import os
import sys
import time
import pickle
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import torch.nn as nn
import pdb

class NesterovAcceleratedGradientOptimizer(Optimizer):
    """
    @brief Follow the Nesterov's implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    """
    def __init__(self, params, lr=required, obj_and_grad_fn=required, constraint_fn=None, use_bb=True,step_size_strategy='bb', backtrack_epsilon=0.95, max_backtrack=10):
        """
        @brief initialization
        @param params variable to optimize
        @param lr learning rate
        @param obj_and_grad_fn a callable function to get objective and gradient
        @param constraint_fn a callable function to force variables to satisfy all the constraints
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution
        # v_k is reference solution
        # obj_k is the objective at v_k
        # a_k is optimization parameter
        # alpha_k is the step size
        # v_k_1 is previous reference solution
        # g_k_1 is gradient to v_k_1
        # obj_k_1 is the objective at v_k_1
        defaults = dict(lr=lr,
                u_k=[], v_k=[], g_k=[], obj_k=[], a_k=[], alpha_k=[],
                v_k_1=[], g_k_1=[], obj_k_1=[],
                v_kp1 = [None],
                obj_eval_count=0)
        super(NesterovAcceleratedGradientOptimizer, self).__init__(params, defaults)
        self.obj_and_grad_fn = obj_and_grad_fn
        self.constraint_fn = constraint_fn
        self.use_bb = use_bb
        self.step_size_strategy = step_size_strategy
        self.backtrack_epsilon = backtrack_epsilon
        self.max_backtrack = max_backtrack

        # I do not know how to get generator's length
        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super(NesterovAcceleratedGradientOptimizer, self).__setstate__(state)

    def step(self, closure=None):
        strategy=self.step_size_strategy.lower()

        if strategy=='nobb':
            return self.step_nobb(closure)
        elif strategy=='bb':
            return self.step_bb(closure)
        elif strategy=='eplace':
            return self.step_eplace(closure)

    def step_nobb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    # directly use p as v_k to save memory
                    #group['v_k'].append(torch.autograd.Variable(p.data, requires_grad=True))
                    group['v_k'].append(p)
                    obj, grad = obj_and_grad_fn(group['v_k'][i])
                    group['g_k'].append(grad.data.clone()) # must clone
                    group['obj_k'].append(obj.data.clone())
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                g_k = group['g_k'][i]
                obj_k = group['obj_k'][i]
                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)
                    obj, grad = obj_and_grad_fn(group['v_k_1'][i])
                    group['g_k_1'].append(grad.data)
                    group['obj_k_1'].append(obj.data.clone())
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                g_k_1 = group['g_k_1'][i]
                obj_k_1 = group['obj_k_1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]

                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]

                # line search with alpha_k as hint
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10

                #ttt = time.time()
                while True:
                    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    u_kp1 = v_k - alpha_k*g_k
                    #constraint_fn(u_kp1)
                    v_kp1.data.copy_(u_kp1 + coef*(u_kp1-u_k))
                    # make sure v_kp1 subjects to constraints
                    # g_kp1 must correspond to v_kp1
                    constraint_fn(v_kp1)

                    f_kp1, g_kp1 = obj_and_grad_fn(v_kp1)

                    #tt = time.time()
                    alpha_kp1 = torch.sqrt(torch.sum((v_kp1.data-v_k.data)**2) / torch.sum((g_kp1.data-g_k.data)**2))
                    # alpha_kp1 = torch.dist(v_kp1.data, v_k.data, p=2) / torch.dist(g_kp1.data, g_k.data, p=2)
                    backtrack_cnt += 1
                    group['obj_eval_count'] += 1
                    #logging.debug("\t\talpha_kp1 %.3f ms" % ((time.time()-tt)*1000))
                    #torch.cuda.synchronize()
                    #logging.debug(prof)

                    #logging.debug("alpha_kp1 = %g, line_search_count = %d, obj_eval_count = %d" % (alpha_kp1, backtrack_cnt, group['obj_eval_count']))
                    #logging.debug("|g_k| = %.6E, |g_kp1| = %.6E" % (g_k.norm(p=2), g_kp1.norm(p=2)))
                    if alpha_kp1 > 0.95*alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        alpha_k.data.copy_(alpha_kp1.data)
                        break
                    else:
                        alpha_k.data.copy_(alpha_kp1.data)
                #if v_k.is_cuda:
                #    torch.cuda.synchronize()
                #logging.debug("\tline search %.3f ms" % ((time.time()-ttt)*1000))

                v_k_1.data.copy_(v_k.data)
                g_k_1.data.copy_(g_k.data)
                obj_k_1.data.copy_(obj_k.data)

                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)

        return loss

    def step_bb(self, closure=None):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            obj_and_grad_fn = self.obj_and_grad_fn
            constraint_fn = self.constraint_fn
            for i, p in enumerate(group['params']):
                #if p.grad is None:
                #    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                obj_k, g_k = obj_and_grad_fn(v_k)
                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i] = obj_k.data.clone()
                if not group['a_k']:
                    group['a_k'].append(torch.ones(1, dtype=g_k.dtype, device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)
                a_k = group['a_k'][i]
                v_k_1 = group['v_k_1'][i]
                obj_k_1, g_k_1 = obj_and_grad_fn(v_k_1)
                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i] = obj_k_1.data.clone()
                if group['v_kp1'][i] is None:
                    group['v_kp1'][i] = torch.autograd.Variable(torch.zeros_like(v_k), requires_grad=True)
                v_kp1 = group['v_kp1'][i]
                if not group['alpha_k']:
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]
                # line search with alpha_k as hint
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                alpha_k = group['alpha_k'][i]
                # line search with alpha_k as hint
                a_kp1 = (1 + (4*a_k.pow(2)+1).sqrt()) / 2
                coef = (a_k-1) / a_kp1
                
                with torch.no_grad():
                    s_k = (v_k - v_k_1)
                    y_k = (g_k - g_k_1)
                    
                    # Numerical stability: check if gradient change is too small
                    y_norm = y_k.norm(p=2)
                    s_norm = s_k.norm(p=2)
                    
                    epsilon = 1e-8  # Small constant to prevent division by zero
                    
                    if y_norm < epsilon:
                        # Gradient is too small, already converged
                        step_size = alpha_k * 0.1  # Use a small fraction of previous step
                    else:
                        # Compute BB step sizes safely
                        s_dot_y = torch.sum(s_k * y_k)
                        y_dot_y = y_k.dot(y_k)
                        
                        # BB short step size (with numerical check)
                        if y_dot_y > epsilon:
                            bb_short_step_size = (s_k.dot(y_k) / y_dot_y).data
                        else:
                            bb_short_step_size = -1.0  # Invalid
                        
                        # Lipschitz step size (with numerical check)
                        lip_step_size = (s_norm / y_norm).data
                        
                        # Choose step size
                        if bb_short_step_size > 0:
                            step_size = bb_short_step_size
                        else:
                            step_size = min(lip_step_size, alpha_k)
                
                # one step
                u_kp1 = v_k - step_size*g_k
                
                # one step
                u_kp1 = v_k - step_size*g_k
                v_kp1.data.copy_(u_kp1 + coef*(u_kp1-u_k))
                constraint_fn(v_kp1)
                group['obj_eval_count'] += 1

                v_k_1.data.copy_(v_k.data)
                #g_k_1.data.copy_(g_k.data)
                #obj_k_1.data.copy_(obj_k.data)
                alpha_k.data.copy_(step_size.data)
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                #g_k.data.copy_(g_kp1.data)
                #obj_k.data.copy_(f_kp1.data)
                a_k.data.copy_(a_kp1.data)

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)
        return loss
    def step_eplace(self,closure=None):
        loss=None
        if closure is not None:
            loss=closure()

        for group in self.param_groups:
            obj_and_grad_fn=self.obj_and_grad_fn
            constraint_fn=self.constraint_fn

            for i,p in enumerate(group['params']):
                # if p.grad is None:  # ← 注释掉
                #     continue        # ← 注释掉
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    group['v_k'].append(p)

                u_k=group['u_k'][i]
                v_k=group['v_k'][i]

                obj_k,g_k=obj_and_grad_fn(v_k)

                if not group['obj_k']:
                    group['obj_k'].append(None)
                group['obj_k'][i]=obj_k.data.clone()

                if not group['a_k']:
                    group['a_k'].append(torch.ones(1,dtype=g_k.dtype,device=g_k.device))
                    group['v_k_1'].append(torch.autograd.Variable(torch.zeros_like(v_k),requires_grad=True))
                    group['v_k_1'][i].data.copy_(group['v_k'][i]-group['lr']*g_k)

                a_k=group['a_k'][i]
                v_k_1=group['v_k_1'][i]

                obj_k_1,g_k_1=obj_and_grad_fn(v_k_1)

                if not group['obj_k_1']:
                    group['obj_k_1'].append(None)
                group['obj_k_1'][i]=obj_k_1.data.clone()

                if group['v_kp1'][i] is None:
                    group['v_kp1'][i]=torch.autograd.Variable(torch.zeros_like(v_k),requires_grad=True)
                v_kp1=group['v_kp1'][i]

                if not group['alpha_k']:
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2)/(g_k-g_k_1).norm(p=2))
                alpha_k=group['alpha_k'][i]

                a_kp1 = (1 + (4 * a_k.pow(2) + 1).sqrt()) / 2
                coef = (a_k - 1) / a_kp1

                with torch.no_grad():
                    s_k = (v_k - v_k_1)
                    y_k = (g_k - g_k_1)
                    
                    # Numerical stability: check if gradient change is too small
                    y_norm = y_k.norm(p=2)
                    s_norm = s_k.norm(p=2)
                    
                    epsilon = 1e-8  # Small constant to prevent division by zero
                    
                    if y_norm < epsilon:
                        # Gradient is too small, already converged, use small step or stop
                        step_size = alpha_k * 0.1  # Use a small fraction of previous step
                    else:
                        # Compute BB step sizes safely
                        s_dot_y = torch.sum(s_k * y_k)
                        y_dot_y = y_k.dot(y_k)
                        
                        # BB short step size (with numerical check)
                        if y_dot_y > epsilon:
                            bb_short_step_size = (s_k.dot(y_k) / y_dot_y).data
                        else:
                            bb_short_step_size = -1.0  # Invalid
                        
                        # Lipschitz step size (with numerical check)
                        if y_norm > epsilon:
                            lip_step_size = (s_norm / y_norm).data
                        else:
                            lip_step_size = alpha_k
                        
                        # Choose step size
                        if bb_short_step_size > 0:
                            step_size = bb_short_step_size
                        else:
                            step_size = min(lip_step_size, alpha_k)
                
                # Backtracking line search (ePlace Algorithm 2)
                backtrack_count = 0
                accepted = False
                
                while backtrack_count < self.max_backtrack and not accepted:
                    # Compute candidate point
                    u_kp1 = v_k - step_size * g_k
                    v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                    constraint_fn(v_kp1)
                    
                    # Evaluate objective at candidate point
                    obj_kp1, g_kp1 = obj_and_grad_fn(v_kp1)
                    group['obj_eval_count'] += 1
                    
                    # Check acceptance criterion (ePlace paper: obj should decrease)
                    # Accept if objective improved sufficiently
                    if obj_kp1.item() <= obj_k.item():
                        accepted = True
                    else:
                        # Backtrack: reduce step size
                        step_size = step_size * self.backtrack_epsilon
                        backtrack_count += 1
                
                # If all backtracks failed, accept the last step anyway
                if not accepted:
                    u_kp1 = v_k - step_size * g_k
                    v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))
                    constraint_fn(v_kp1)
                
                # Update stored values
                                
                v_k_1.data.copy_(v_k.data)
                alpha_k.data.copy_(step_size.data if isinstance(step_size, torch.Tensor) else torch.tensor(step_size, dtype=alpha_k.dtype, device=alpha_k.device))
                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                a_k.data.copy_(a_kp1.data)
                
                # Ensure parameter is updated (important!)
                p.data.copy_(v_k.data)
        
        return loss
                    