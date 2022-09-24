import torch
import numpy as np
import torch.nn as nn
import sklearn.preprocessing
import torch.nn.functional as F

class HierarchicalProxyLoss(nn.Module):
    def __init__(self, num_classes, embedding_size=128, scale=3, proxy_per_class=5, sub_proxy_reduction='min', w1=1, w2=1):
        super(HierarchicalProxyLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.proxy_per_class = proxy_per_class
        self.reduction = sub_proxy_reduction
        self.w1 = w1
        self.w2 = w2

        self.main_proxy = nn.Parameter(torch.randn(1, num_classes, embedding_size) / 8)
        self.sub_proxy = nn.Parameter(torch.randn(proxy_per_class, num_classes, embedding_size) / 8)

        # just for debuging
        self.iter = 0

    def cast_types(self, dtype, device):
        self.proxies.data = self.proxies.data.to(device).type(dtype)

    def proxy_vector_loss(self, embeddings, labels, proxies, reduction=None):
        proxy_vector_distance = []
        for proxy in proxies:
            proxy = self.scale * F.normalize(proxy, p=2, dim=-1)
            embeddings = self.scale * F.normalize(embeddings, p=2, dim=-1)
            distance = pairwise_distance(embeddings, proxy, squared=True)

            proxy_vector_distance.append(distance.unsqueeze(0))

        proxy_vector_distance = torch.cat(proxy_vector_distance, dim=0)
        if reduction == 'max':
            proxy_vector_distance, _ = torch.max(proxy_vector_distance, dim=0)
        elif reduction == 'min':
            proxy_vector_distance, _ = torch.min(proxy_vector_distance, dim=0)
        elif reduction == None:
            proxy_vector_distance = proxy_vector_distance[0]

        T = binarize_and_smooth_labels(
            T=labels, nb_classes=self.num_classes, smoothing_const=0
        )
        loss1 = torch.sum(T * torch.exp(-proxy_vector_distance), -1)
        loss2 = torch.sum((1 - T) * torch.exp(-proxy_vector_distance), -1)
        loss = -torch.log(loss1 / loss2)
        loss = loss.mean()

        return loss

    def proxy_proxy_loss(self, main_proxy, sub_proxy):
        total_loss = 0
        for proxy in sub_proxy:
            proxy = self.scale * F.normalize(proxy, p=2, dim=-1)
            main_proxy = self.scale * F.normalize(main_proxy, p=2, dim=-1)
            distance = pairwise_distance(main_proxy[0], proxy, squared=True)

            labels = torch.Tensor([i for i in range(self.num_classes)])
            T = binarize_and_smooth_labels(
                T=labels, nb_classes=self.num_classes, smoothing_const=0
            )

            loss1 = torch.sum(T * torch.exp(-distance), -1)
            loss2 = torch.sum((1 - T) * torch.exp(-distance), -1)
            loss = -torch.log(loss1 / loss2)
            loss = loss.mean()
            total_loss += loss

        return total_loss / self.proxy_per_class

    def forward(self, embeddings, labels, indices_tuple):
        main_proxy_loss = self.proxy_vector_loss(embeddings, labels, self.main_proxy, reduction=None)
        sub_proxy_loss = self.proxy_vector_loss(embeddings, labels, self.sub_proxy, reduction=self.reduction)
        proxy_loss = self.proxy_proxy_loss(self.main_proxy, self.sub_proxy)
        return main_proxy_loss + self.w1 * sub_proxy_loss + self.w2 * proxy_loss


def pairwise_distance(a, b, squared=False):
    pairwise_distances_squared = -2 * torch.mm(a, torch.t(b))
    a = a.pow(2).sum(dim=1, keepdim=True)
    b = b.pow(2).sum(dim=1, keepdim=False).unsqueeze(0)
    pairwise_distances_squared += a + b

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )
    return pairwise_distances


def binarize_and_smooth_labels(T, nb_classes, smoothing_const=0):

    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T
