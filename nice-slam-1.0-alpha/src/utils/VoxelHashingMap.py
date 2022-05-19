import torch
import einops

class VoxelHashingMap(object):
    def __init__(self,grid_resolution,init_size,feature_len,bound_min,voxel_size,name):
        """
        :param grid_resolution (3, ) resolution in three directions (x,y,z)
        :param init_size (scalar) initial size of the voxels
        :param feature len (scalar) length of the feature
        :param bound_min (double) tensor (3,)
        :param voxel_size: double
        """
        self.vox_idx = -torch.ones(grid_resolution[0] * grid_resolution[1] * grid_resolution[2]).long()
        self.voxels = torch.zeros(init_size,feature_len).normal_(mean=0, std=0.01)
        self.vox_pos = torch.zeros(init_size,dtype=torch.long)
        self.n_xyz = grid_resolution
        self.n_occupied = torch.tensor(0).long()
        self.latent_dim = feature_len
        self.voxel_size = voxel_size
        self.bound_min = bound_min
        self.bound_max = self.bound_min + self.voxel_size * torch.tensor(self.n_xyz)
        self.device = 'cpu'
        self.name = name
        self.tracker = False
    
    def print_info(self):
        print("Content of map",self.name,":")
        print(" vox_idx.shape:",self.vox_idx.shape)
        print(" vox_idx:",self.vox_idx)
        print(" vox_idx max:",self.vox_idx.max())
        print(" vox_idx min:",self.vox_idx.min())
        print(" voxels.shape:",self.voxels.shape)
        print(" voxels:",self.voxels)
        print(" vox_pos.shape:",self.vox_pos.shape)
        print(" vox_pos:",self.vox_pos)
        print(" n_xyz:",self.n_xyz)
        print(" n_occupied:",self.n_occupied)
        print(" latent_dim",self.latent_dim)
        print(" voxel_size",self.voxel_size)
        print(" bound_min",self.bound_min)
        print(" bound_max",self.bound_max)
        print(" device:",self.device)                
    
    def id3d_to_id1d(self, xyz: torch.Tensor):
        """
        :param xyz (N, 3) long id
        :return: (N, ) lineraized id to be accessed in self.indexer
        """
        ret = xyz[:, 2] + self.n_xyz[-1] * xyz[:, 1] + (self.n_xyz[-1] * self.n_xyz[-2]) * xyz[:, 0]
        return ret.long()
        
    def id1d_to_id3d(self, idx: torch.Tensor):
        """
        :param idx: (N, ) linearized id for access in self.indexer
        :return: xyz (N, 3) id to be indexed in 3D
        """
        ret = torch.stack([idx // (self.n_xyz[1] * self.n_xyz[2]),
                        (idx // self.n_xyz[2]) % self.n_xyz[1],
                        idx % self.n_xyz[2]], dim=-1)
        return ret.long()
    
    def point3d_to_id3d(self, xyz: torch.Tensor):
        xyz_zeroed = xyz - self.bound_min.unsqueeze(0)
        xyz_normalized = xyz_zeroed / self.voxel_size
        grid_id = torch.floor(xyz_normalized).long()
        return grid_id

    def point3d_to_id1d(self, xyz: torch.Tensor):
        xyz_zeroed = xyz - self.bound_min.unsqueeze(0)
        xyz_normalized = xyz_zeroed / self.voxel_size
        grid_id = torch.floor(xyz_normalized).long()
        grid_id = self._linearize_id(grid_id)
        return grid_id
    
    def id1d_to_point3d(self, idx: torch.Tensor):
        n_xyz_id = self._unlinearize_id(idx)
        corner_xyz = n_xyz_id*self.voxel_size+self.bound_min
        return corner_xyz
    
    def id3d_to_point3d(self, n_xyz: torch.Tensor):
        corner_xyz = n_xyz*self.voxel_size+self.bound_min
        return corner_xyz

    def find_neighbors(self, points:torch.Tensor):
        """
        :param points: (N, 3) 3d coordinates of target points
        :return: 8 neighbors for each points (id1d) (N(8-m),)
        """
        neighbor = torch.Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        neighbor = neighbor.to(self.device)
        voxel_xyz_id = self.point3d_to_id3d(points)
        
        voxel_xyz_id = voxel_xyz_id.repeat(1, 8).reshape(voxel_xyz_id.shape[0] * 8, voxel_xyz_id.shape[1])
        neighbor = neighbor.repeat(points.shape[0], 1)
        res = voxel_xyz_id + neighbor

        mask_x = (res[:, 0] < self.n_xyz[0]) & (res[:, 0] >= 0)
        mask_y = (res[:, 1] < self.n_xyz[1]) & (res[:, 1] >= 0)
        mask_z = (res[:, 2] < self.n_xyz[2]) & (res[:, 2] >= 0)
        valid_mask = mask_x & mask_y & mask_z
        res = self.id3d_to_id1d(res[valid_mask])
        # fuck this bullshit!!!
        return torch.unique(res)

    def find_neighbors_feature(self, points:torch.Tensor):
        """
        :param points: (N, 3) 3d coordinates of target pointss
        :return: 8 neighbors features for each points  (8N, dim)
        """
        neighbor = torch.Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1] , [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        neighbor = neighbor.to(self.device)
        voxel_xyz_id = self.point3d_to_id3d(points)
        voxel_xyz_id = voxel_xyz_id.repeat(1, 8).reshape(voxel_xyz_id.shape[0] * 8, voxel_xyz_id.shape[1])
        neighbor = neighbor.repeat(points.shape[0], 1)
        idx = voxel_xyz_id + neighbor
        mask_x = (idx[:, 0] < self.n_xyz[0]) & (idx[:, 0] >= 0)
        mask_y = (idx[:, 1] < self.n_xyz[1]) & (idx[:, 1] >= 0)
        mask_z = (idx[:, 2] < self.n_xyz[2]) & (idx[:, 2] >= 0)
        valid_mask = (mask_x & mask_y & mask_z)
        coordinate3d = self.id3d_to_point3d(idx)

        grid_idx = self.id3d_to_id1d(idx[valid_mask])

        hashmap_idx = self.vox_idx[grid_idx]    
        existence_mask = (hashmap_idx != -1).clone()
        valid_mask[valid_mask.clone()] = existence_mask
        
        res_features = torch.zeros([points.shape[0]*8, self.latent_dim]).to(self.device)
        res_features[valid_mask]= self.voxels[hashmap_idx[existence_mask]]
        res_features = res_features.to(self.device)
        return res_features, coordinate3d, valid_mask.reshape((points.shape[0], 8))
    
    def if_invalid_allocate(self,neighbors:torch.Tensor):
        neighbors_vox_idx = self.vox_idx[neighbors]
        invalid_neighbors = neighbors[neighbors_vox_idx == -1] 
        to_be_allocate_amount = invalid_neighbors.shape[0]
        self.allocate_blocks(to_be_allocate_amount)
        for i in range(to_be_allocate_amount):
            self.vox_idx[invalid_neighbors[i]] = self.n_occupied+i
            self.vox_pos[self.n_occupied+i] = invalid_neighbors[i]
        self.n_occupied += to_be_allocate_amount        

    def allocate_blocks(self,count:int):

        # allocate more spaces if access new voxels
        target_n_occupied = self.n_occupied + count
        # allocate new slots
        if self.voxels.shape[0] < target_n_occupied:
            new_size = self.voxels.shape[0]
            while new_size < target_n_occupied:
                new_size *= 2
            new_voxels = torch.empty((new_size, self.latent_dim), dtype=torch.float32)
            new_voxels[:self.voxels.shape[0]] = self.voxels
            new_vox_pos = torch.ones((new_size, ), dtype=torch.long) * -1
            new_vox_pos[:self.voxels.shape[0]] = self.vox_pos

            new_voxels[self.voxels.shape[0]:].zero_().normal_(mean=0, std=0.01)
            self.voxels = new_voxels
            self.vox_pos = new_vox_pos
            
            self.voxels.share_memory_()
            self.vox_pos.share_memory_()
        if self.device is not 'cpu':
            self.voxels = self.voxels.to(self.device)
        
    def get_feature_at(self,coordinate:torch.Tensor):
        """
        :param coordinate (N, 3) long id
        :return: (N, dim) corresponding feature, invaild coordinate is 0
        """
        mask_x = (coordinate[:, 0] < self.bound_max[0]) & (coordinate[:, 0] > self.bound_min[0])
        mask_y = (coordinate[:, 1] < self.bound_max[1]) & (coordinate[:, 1] > self.bound_min[1])
        mask_z = (coordinate[:, 2] < self.bound_max[2]) & (coordinate[:, 2] > self.bound_min[2])
        valid_mask = mask_x & mask_y & mask_z
        idx = self.id3d_to_id1d(coordinate[valid_mask])
        ret = torch.zeros(coordinate.size(0),self.latent_dim)
        ret[valid_mask] = self.voxels[idx]
        return ret
    
    def map_interpolation(self, points:torch.Tensor):
        if points.shape[0] == 0:
            return torch.Tensor([]).reshape([0,self.latent_dim]).float().to(self.device)
        # (N*8, dim), (N*8, 3), (N*8, )
        # [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1] , [1, 0, 1], [0, 1, 1], [1, 1, 1]
        neighbors_feature, neighbors_coordinate, valid_mask = self.find_neighbors_feature(points)
        # (N, 4, 2, dim)
        neighbors_feature = neighbors_feature.reshape([points.shape[0], 4, 2, -1]) 
        # (N, 4, 2, 3)
        neighbors_coordinate = neighbors_coordinate.reshape([points.shape[0], 4, 2, -1])
        # (N, 4, 2)
        valid_mask = valid_mask.reshape([points.shape[0], 4, 2])
        
        points_1 = neighbors_coordinate[:, 3, 1, :]
        points_0 = neighbors_coordinate[:, 0, 0, :]
        points_d = (points - points_0)/(points_1 - points_0)
        
        points_weight = torch.stack([1-points_d, points_d], axis = 2).float()
        
        # (N, 2)
        x_weight, y_weight, z_weight = points_weight[:, 0], points_weight[:, 1], points_weight[:, 2]
        
        # First interpolation
        x_weight = einops.repeat(x_weight, 'm n -> m k n', k=4)
        
        x_weight = x_weight * valid_mask
        x_weight = x_weight / x_weight.sum(axis = -1)[:,:, None]
        x_weight = torch.nan_to_num(x_weight,nan=0.0).float()
        
        neighbors_feature_1 = torch.einsum("ijkl,ijk->ijl", neighbors_feature, x_weight)
        
        # Second interpolation
        y_weight = einops.repeat(y_weight, 'm n -> m k n', k=2)
        
        valid_mask_y = (valid_mask.sum(axis = -1) != 0).reshape(points.shape[0], 2, 2)
        y_weight = y_weight * valid_mask_y
        y_weight = y_weight / y_weight.sum(axis = -1)[:,:, None]
        y_weight = torch.nan_to_num(y_weight,nan=0.0).float()
        
        neighbors_feature_1 = neighbors_feature_1.reshape([points.shape[0], 2, 2, -1]) 
        neighbors_feature_2 = torch.einsum("ijkl,ijk->ijl", neighbors_feature_1, y_weight)
        
        # Third interpolation
        valid_mask_z = (valid_mask_y.sum(axis = -1) != 0)
        
        z_weight = z_weight * valid_mask_z
        z_weight = z_weight / z_weight.sum(axis = -1)[:, None]
        z_weight = torch.nan_to_num(z_weight,nan=0.0).float()
        
        neighbors_feature_3 = torch.einsum("ikl,ik->il", neighbors_feature_2, z_weight)
        return neighbors_feature_3
    
    # def map_interpolation(self, points:torch.Tensor):
    #     if points.shape[0] == 0:
    #         return torch.Tensor([]).reshape([0,self.latent_dim]).float().to(self.device)
    #     # (N*8, dim), (N*8, 3), (N*8, )
    #     # [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1] , [1, 0, 1], [0, 1, 1], [1, 1, 1]
    #     neighbors_feature, neighbors_coordinate, valid_mask = self.find_neighbors_feature(points)
    #     # (N, 4, 2, dim)
    #     neighbors_feature = neighbors_feature.reshape([points.shape[0], 4, 2, -1]) 
    #     # (N, 4, 2, 3)
    #     neighbors_coordinate = neighbors_coordinate.reshape([points.shape[0], 4, 2, -1])
    #     points_1 = neighbors_coordinate[:, 3, 1, :]
    #     points_0 = neighbors_coordinate[:, 0, 0, :]
    #     points_d = (points - points_0)/(points_1 - points_0)
    #     points_weight = torch.stack([1-points_d, points_d], axis = 2).float()
    #     x_weight, y_weight, z_weight = points_weight[:, 0], points_weight[:, 1], points_weight[:, 2]
    #     # First interpolation
    #     neighbors_feature_1 = torch.einsum("ijkl,ik->ijl", neighbors_feature, x_weight)
    #     # Second interpolation
    #     neighbors_feature_1 = neighbors_feature_1.reshape([points.shape[0], 2, 2, -1]) 
    #     neighbors_feature_2 = torch.einsum("ijkl,ik->ijl", neighbors_feature_1, y_weight)
    #     # Third interpolation
    #     neighbors_feature_3 = torch.einsum("ikl,ik->il", neighbors_feature_2, z_weight)
    #     return neighbors_feature_3
    
    # def map_interpolation(self, points:torch.Tensor):
        
    #     if points.shape[0] == 0:
    #         return torch.Tensor([]).reshape([0,self.latent_dim]).float().to(self.device)
    #     # N*8*dim
    #     neighbors_feature, neighbors_coordinate, valid_mask = self.find_neighbors_feature(points)
    #     neighbors_feature = neighbors_feature.reshape([points.shape[0],8,-1]) 
    #     neighbors_coordinate = neighbors_coordinate.reshape([points.shape[0],8,-1])
    #     # N*8*1 = (N*8*3, N*1*3) 
    #     distances = torch.cdist(neighbors_coordinate.float(), points[:,None,:].float(), p=2)
    #     # N*8
    #     distances = distances.squeeze(-1)
    #     weight = 1.0/distances
    #     weight = weight / torch.sum(weight, axis = 1)[:, None]
    #     weight = torch.nan_to_num(weight,nan=1.0).float()
    #     weight *= valid_mask
    #     assert((weight.sum(axis=1)==0).sum() == 0)
    #     weight = weight / torch.sum(weight, axis = 1)[:, None]
    #     return torch.einsum("ijk,ij->ik",neighbors_feature,weight)
    
    def get_feature_by_id3d_mask(self,mask):
        
        id1d_mask = mask.reshape(self.n_xyz[0]*self.n_xyz[1]*self.n_xyz[2])
        

        id1d = id1d_mask.nonzero().squeeze(-1)
        self.if_invalid_allocate(id1d)
        
        return self.voxels[self.vox_idx[id1d]]
    
    def put_feature_by_id3d_mask(self,mask,feature):
        id1d_mask = mask.reshape(self.n_xyz[0]*self.n_xyz[1]*self.n_xyz[2])
        id1d = id1d_mask.nonzero().squeeze(-1)
        self.voxels[self.vox_idx[id1d]] = feature

    def to(self, device = 'cuda:0'):
        self.voxels = self.voxels.to(device)
        self.vox_idx = self.vox_idx.to(device)
        self.vox_pos = self.vox_pos.to(device)
        self.bound_min = self.bound_min.to(device)
        self.bound_max = self.bound_max.to(device)  
        self.device = device
        return self
    
    def detach(self):
        self.voxels = self.voxels.detach()
        return self
    
    def share_memory_(self):
        self.vox_idx.share_memory_()
        self.voxels.share_memory_()
        self.vox_pos.share_memory_()
        self.bound_min.share_memory_()
        self.bound_max.share_memory_() 
        self.n_occupied.share_memory_()
    
    def clone(self):
        new_hash_map = VoxelHashingMap([3,3,3],0,3,torch.tensor([1,2,3]),1, "temp")
        new_hash_map.vox_idx = self.vox_idx.clone()
        new_hash_map.voxels = self.voxels.clone()
        new_hash_map.vox_pos = self.vox_pos.clone()
        new_hash_map.bound_min = self.bound_min.clone()
        new_hash_map.bound_max = self.bound_max.clone()
        new_hash_map.n_occupied = self.n_occupied.clone()
        new_hash_map.n_xyz = self.n_xyz
        new_hash_map.latent_dim = self.latent_dim
        new_hash_map.voxel_size = self.voxel_size
        new_hash_map.name = self.name
        return new_hash_map
    
    def if_neighbors_valid(self, points:torch.Tensor):
        """
        :param points: (N, 3) 3d coordinates of target points
        :return: mask (N,) indicates whether each point has at least one valid neighbor
        """
        neighbor = torch.Tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
        neighbor = neighbor.to(self.device)
        voxel_xyz_id = self.point3d_to_id3d(points)
        
        voxel_xyz_id = voxel_xyz_id.repeat(1, 8).reshape(voxel_xyz_id.shape[0] * 8, voxel_xyz_id.shape[1])
        neighbor = neighbor.repeat(points.shape[0], 1)
        idx = voxel_xyz_id + neighbor
        
        mask_x = (idx[:, 0] < self.n_xyz[0]) & (idx[:, 0] >= 0)
        mask_y = (idx[:, 1] < self.n_xyz[1]) & (idx[:, 1] >= 0)
        mask_z = (idx[:, 2] < self.n_xyz[2]) & (idx[:, 2] >= 0)
        bound_mask = (mask_x & mask_y & mask_z).to(self.device)
        
        neighbors_valid_mask = torch.zeros(points.shape[0]*8,dtype=torch.bool).to(self.device)
                
        grid_idx = self.id3d_to_id1d(idx[bound_mask])
        hash_idx = self.vox_idx[grid_idx]
        neighbors_valid_mask[bound_mask] = (hash_idx!=-1)
        neighbors_valid_mask = neighbors_valid_mask.reshape(points.shape[0],8)
        ret_mask = neighbors_valid_mask.sum(dim=1)!=0
        return ret_mask