import torch
import torch.nn as nn

class Subspace_Projection(nn.Module):
    def __init__(self, num_dim=5):
        super().__init__()
        self.num_dim = num_dim

    def create_subspace(self, supportset_features, class_size, sample_size):
        all_hyper_planes = []
        means = []
        for ii in range(class_size):
            num_sample = sample_size
            all_support_within_class_t = supportset_features[ii]
            meann = torch.mean(all_support_within_class_t, dim=0)
            means.append(meann)
            all_support_within_class_t = all_support_within_class_t - meann.unsqueeze(0).repeat(num_sample, 1)
            all_support_within_class = torch.transpose(all_support_within_class_t, 0, 1)
            uu, s, v = torch.svd(all_support_within_class.double(), some=False)
            uu = uu.float()
            all_hyper_planes.append(uu[:, :self.num_dim])

        all_hyper_planes = torch.stack(all_hyper_planes, dim=0)
        means = torch.stack(means)

        if len(all_hyper_planes.size()) < 3:
            all_hyper_planes = all_hyper_planes.unsqueeze(-1)

        return all_hyper_planes, means


    def projection_metric(self, target_features, hyperplanes, mu):
        eps = 1e-12
        batch_size = target_features.shape[0]
        class_size = hyperplanes.shape[0]

        similarities = []

        discriminative_loss = 0.0

        for j in range(class_size):
            h_plane_j =  hyperplanes[j].unsqueeze(0).repeat(batch_size, 1, 1)
            target_features_expanded = (target_features - mu[j].expand_as(target_features)).unsqueeze(-1)
            projected_query_j = torch.bmm(h_plane_j, torch.bmm(torch.transpose(h_plane_j, 1, 2), target_features_expanded))
            projected_query_j = torch.squeeze(projected_query_j) + mu[j].unsqueeze(0).repeat(batch_size, 1)
            projected_query_dist_inter = target_features - projected_query_j

            #Training per epoch is slower but less epochs in total
            query_loss = -torch.sqrt(torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) + eps) # norm ||.||

            #Training per epoch is faster but more epochs in total
            #query_loss = -torch.sum(projected_query_dist_inter * projected_query_dist_inter, dim=-1) # Squared norm ||.||^2

            similarities.append(query_loss)

            for k in range(class_size):
                if j != k:
                   temp_loss = torch.mm(torch.transpose(hyperplanes[j], 0, 1), hyperplanes[k]) ## discriminative subspaces (Conv4 only, ResNet12 is computationally expensive)
                   discriminative_loss = discriminative_loss + torch.sum(temp_loss*temp_loss)

        similarities = torch.stack(similarities, dim=1)

        return similarities, discriminative_loss
