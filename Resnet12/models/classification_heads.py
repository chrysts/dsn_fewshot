import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
#from qpth.qp import QPFunction
import torch.nn.functional as F


def sqrt_newton_schulz(A, numIters):
    dim = A.shape[0]
    normA = A.mul(A).sum(dim=0).sum(dim=0).sqrt()
    Y = A.div(normA.expand_as(A))
    I = torch.eye(dim, dim).float().cuda()
    Z = torch.eye(dim, dim).float().cuda()
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.mm(Y))
        Y = Y.mm(T)
        Z = T.mm(Z)

    # sA = Y * torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)

    # sA = Y * torch.sqrt(normA).expand_as(A)

    sZ = Z * 1. / torch.sqrt(normA).expand_as(A)
    return sZ


def polar_decompose(input):
    # square_mat = input.mm(input.transpose(0, 1))
    # square_mat = square_mat/torch.norm(torch.diag(square_mat), p=1)
    # ortho_mat = self.sqrt_newton_schulz(square_mat, numIters=1)

    square_mat = input.transpose(0, 1).mm(input)
    sA_minushalf = sqrt_newton_schulz(square_mat, 1)
    ortho_mat = input.mm(sA_minushalf)

    # return ortho_mat

    return ortho_mat.mm(ortho_mat.transpose(0, 1))


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))


def binv(b_mat):
    """
    Computes an inverse of each matrix in the batch.
    Pytorch 0.4.1 does not support batched matrix inverse.
    Hence, we are solving AX=I.
    
    Parameters:
      b_mat:  a (n_batch, n, n) Tensor.
    Returns: a (n_batch, n, n) Tensor.
    """

    id_matrix = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat).cuda()
    b_inv, _ = torch.gesv(id_matrix, b_mat)
    
    return b_inv


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))




#################  uncomment this if you have installed QPFunction and run Ridge
# def MetaOptNetHead_Ridge(query, support, support_labels, n_way, n_shot, lambda_reg=50.0, double_precision=True):
#     """
#     Fits the support set with ridge regression and
#     returns the classification score on the query set.
#
#     Parameters:
#       query:  a (tasks_per_batch, n_query, d) Tensor.
#       support:  a (tasks_per_batch, n_support, d) Tensor.
#       support_labels: a (tasks_per_batch, n_support) Tensor.
#       n_way: a scalar. Represents the number of classes in a few-shot classification task.
#       n_shot: a scalar. Represents the number of support examples given per class.
#       lambda_reg: a scalar. Represents the strength of L2 regularization.
#     Returns: a (tasks_per_batch, n_query, n_way) Tensor.
#     """
#
#     tasks_per_batch = query.size(0)
#     n_support = support.size(1)
#     n_query = query.size(1)
#
#     assert(query.dim() == 3)
#     assert(support.dim() == 3)
#     assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
#     assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
#
#     #Here we solve the dual problem:
#     #Note that the classes are indexed by m & samples are indexed by i.
#     #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
#
#     #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
#
#     #\alpha is an (n_support, n_way) matrix
#     kernel_matrix = computeGramMatrix(support, support)
#     kernel_matrix += lambda_reg * torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
#
#     block_kernel_matrix = kernel_matrix.repeat(n_way, 1, 1) #(n_way * tasks_per_batch, n_support, n_support)
#
#     support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_way)
#     support_labels_one_hot = support_labels_one_hot.transpose(0, 1) # (n_way, tasks_per_batch * n_support)
#     support_labels_one_hot = support_labels_one_hot.reshape(n_way * tasks_per_batch, n_support)     # (n_way*tasks_per_batch, n_support)
#
#     G = block_kernel_matrix
#     e = -2.0 * support_labels_one_hot
#
#     #This is a fake inequlity constraint as qpth does not support QP without an inequality constraint.
#     id_matrix_1 = torch.zeros(tasks_per_batch*n_way, n_support, n_support)
#     C = Variable(id_matrix_1)
#     h = Variable(torch.zeros((tasks_per_batch*n_way, n_support)))
#     dummy = Variable(torch.Tensor()).cuda()      # We want to ignore the equality constraint.
#
#     #if double_precision:
#     G, e, C, h = [x.double().cuda() for x in [G, e, C, h]]
#
#
#     qp_sol = QPFunction(verbose=False)(G, e.detach(), C.detach(), h.detach(), dummy.detach(), dummy.detach())
#     qp_sol = qp_sol.reshape(n_way, tasks_per_batch, n_support)
#     qp_sol = qp_sol.permute(1, 2, 0)
#
#
#     # Compute the classification score.
#     compatibility = computeGramMatrix(support, query)
#     compatibility = compatibility.float()
#     compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
#     qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
#     logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
#     logits = logits * compatibility
#     logits = torch.sum(logits, 1)
#
#     return logits

def R2D2Head(query, support, support_labels, n_way, n_shot, l2_regularizer_lambda=50.0):
    """
    Fits the support set with ridge regression and 
    returns the classification score on the query set.
    
    This model is the classification head described in:
    Meta-learning with differentiable closed-form solvers
    (Bertinetto et al., in submission to NIPS 2018).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      l2_regularizer_lambda: a scalar. Represents the strength of L2 regularization.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    id_matrix = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()
    
    # Compute the dual form solution of the ridge regression.
    # W = X^T(X X^T - lambda * I)^(-1) Y
    ridge_sol = computeGramMatrix(support, support) + l2_regularizer_lambda * id_matrix
    ridge_sol = binv(ridge_sol)
    ridge_sol = torch.bmm(support.transpose(1,2), ridge_sol)
    ridge_sol = torch.bmm(ridge_sol, support_labels_one_hot)
    
    # Compute the classification score.
    # score = W X
    logits = torch.bmm(query, ridge_sol)

    return logits



def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and 
    returns the classification score (=L2 distance to each class prototype) on the query set.
    
    This model is the classification head described in:
    Prototypical Networks for Few-shot Learning
    (Snell et al., NIPS 2017).
    
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)
    
    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    
    # From:
    # https://github.com/gidariss/FewShotWithoutForgetting/blob/master/architectures/PrototypicalNetworksHead.py
    #************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1,2)

    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits
    
    if normalize:
        logits = logits / d

    return logits



def SubspaceNetHead(query, support, support_labels, n_way, n_shot, normalize=True):
    """
       Constructs the subspace representation of each class(=mean of support vectors of each class) and
       returns the classification score (=L2 distance to each class prototype) on the query set.

        Our algorithm using subspaces here

       Parameters:
         query:  a (tasks_per_batch, n_query, d) Tensor.
         support:  a (tasks_per_batch, n_support, d) Tensor.
         support_labels: a (tasks_per_batch, n_support) Tensor.
         n_way: a scalar. Represents the number of classes in a few-shot classification task.
         n_shot: a scalar. Represents the number of support examples given per class.
         normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
       Returns: a (tasks_per_batch, n_query, n_way) Tensor.
       """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way)
    #support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)


    support_reshape = support.view(tasks_per_batch * n_support, -1)

    support_labels_reshaped = support_labels.contiguous().view(-1)
    class_representatives = []
    for nn in range(n_way):
        idxss = (support_labels_reshaped == nn).nonzero()
        all_support_perclass = support_reshape[idxss, :]
        class_representatives.append(all_support_perclass.view(tasks_per_batch, n_shot, -1))

    class_representatives = torch.stack(class_representatives)
    class_representatives = class_representatives.transpose(0, 1) #tasks_per_batch, n_way, n_support, -1
    class_representatives = class_representatives.transpose(2, 3).contiguous().view(tasks_per_batch*n_way, -1, n_shot)

    dist = []
    for cc in range(tasks_per_batch*n_way):
        batch_idx = cc//n_way
        qq = query[batch_idx]
        uu, _, _ = torch.svd(class_representatives[cc].double())
        uu = uu.float()
        subspace = uu[:, :n_shot-1].transpose(0, 1)
        projection = subspace.transpose(0, 1).mm(subspace.mm(qq.transpose(0, 1))).transpose(0, 1)
        dist_perclass = torch.sum((qq - projection)**2, dim=-1)
        dist.append(dist_perclass)

    dist = torch.stack(dist).view(tasks_per_batch, n_way, -1).transpose(1, 2)
    logits = -dist

    if normalize:
        logits = logits / d

    return logits




class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=True):
        super(ClassificationHead, self).__init__()
        if ('Subspace' in base_learner):
            self.head = SubspaceNetHead
        elif ('Ridge' in base_learner):
            self.head = MetaOptNetHead_Ridge
        elif ('R2D2' in base_learner):
            self.head = R2D2Head
        elif ('Proto' in base_learner):
            self.head = ProtoNetHead
        else:
            print ("Cannot recognize the base learner type")
            assert(False)
        
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))
        
    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)
