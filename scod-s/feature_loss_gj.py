import torch
import torch.nn.functional as F

"""
The following code is modified from the file in https://github.com/siyueyu/SCWSSOD/blob/f8650567cbbc8df5bf6edc32a633c47a885574cd/lscloss.py.
Credit for them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 保持原有的 FeatureLoss 基础逻辑不变，增加结构引导功能
class StructureGuidedALSC(nn.Module):
    def __init__(self):
        super(StructureGuidedALSC, self).__init__()
        # 定义拉普拉斯核 (二阶差分)，用于提取结构边界
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        self.register_buffer('laplacian_kernel', kernel.view(1, 1, 3, 3))

    def get_semantic_boundary(self, f1_feature):
        """
        从 Backbone 的浅层特征 f1_feature 提取语义边界
        输入: [B, C, H, W]
        输出: [B, 1, H, W] 的归一化边界图
        """
        # 1. 降维: 使用 Max Pooling 聚合通道信息，捕捉最强烈的纹理响应
        f1_compressed, _ = torch.max(f1_feature, dim=1, keepdim=True) # [B, 1, H, W]
        
        # 2. 归一化: 确保特征在做差分前处于统一尺度 (Instance Norm)
        f1_norm = F.instance_norm(f1_compressed)
        
        # 3. 计算二阶差分 (Laplacian)
        # Padding=1 保持尺寸不变
        boundary_response = F.conv2d(f1_norm, self.laplacian_kernel, padding=1)
        
        # 4. 取绝对值 (Magnitude) 并 Detach (阻断梯度)
        boundary_mag = torch.abs(boundary_response).detach()
        
        return boundary_mag

    def _normalize_map(self, feat_map):
        """将特征图归一化到 [0, 1] 区间，防止 sigma 失效"""
        B, C, H, W = feat_map.shape
        feat_flat = feat_map.view(B, -1)
        min_v = feat_flat.min(dim=1, keepdim=True)[0]
        max_v = feat_flat.max(dim=1, keepdim=True)[0]
        norm_map = (feat_flat - min_v) / (max_v - min_v + 1e-6)
        return norm_map.view(B, C, H, W)

    def forward(self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
                structure_feature=None, custom_modality_downsamplers=None, out_kernels_vis=False):
        
        # 如果传入了 structure_feature (来自 backbone 的 F1), 则计算结构图并加入 sample
        if structure_feature is not None:
            # 1. 计算边界图
            struct_map = self.get_semantic_boundary(structure_feature)
            # 2. 上采样到与输入图像一致
            struct_map = F.interpolate(struct_map, size=(height_input, width_input), mode='bilinear', align_corners=False)
            # 3. 再次归一化，确保数值范围适配 sigma
            struct_map = self._normalize_map(struct_map)
            # 4. 加入 sample 字典
            sample['structure'] = struct_map

        # 复用原有的核心计算逻辑 (_create_kernels, _unfold 等)
        # 注意：这里直接调用原 FeatureLoss 的逻辑，或者将原 FeatureLoss 的方法复制进来
        # 为了简洁，假设你保留了原 FeatureLoss 的静态方法作为工具函数
        return FeatureLoss.forward(self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, 
                                   custom_modality_downsamplers=custom_modality_downsamplers, out_kernels_vis=out_kernels_vis)


class FeatureLoss(torch.nn.Module):
    """
    This loss function based on the following paper.
    Please consider using the following bibtex for citation:
    @article{obukhov2019gated,
        author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
        title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
        journal={CoRR},
        volume={abs/1906.04651},
        year={2019},
        url={http://arxiv.org/abs/1906.04651},
    }
    """
    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        """
        Performs the forward pass of the loss.
        :param y_hat_softmax: A tensor of predicted per-pixel class probabilities of size NxCxHxW
        :param kernels_desc: A list of dictionaries, each describing one Gaussian kernel composition from modalities.
            The final kernel is a weighted sum of individual kernels. Following example is a composition of
            RGBXY and XY kernels:
            kernels_desc: [{
                'weight': 0.9,          # Weight of RGBXY kernel
                'xy': 6,                # Sigma for XY
                'rgb': 0.1,             # Sigma for RGB
            },{
                'weight': 0.1,          # Weight of XY kernel
                'xy': 6,                # Sigma for XY
            }]
        :param kernels_radius: Defines size of bounding box region around each pixel in which the kernel is constructed.
        :param sample: A dictionary with modalities (except 'xy') used in kernels_desc parameter. Each of the provided
            modalities is allowed to be larger than the shape of y_hat_softmax, in such case downsampling will be
            invoked. Default downsampling method is area resize; this can be overriden by setting.
            custom_modality_downsamplers parameter.
        :param width_input, height_input: Dimensions of the full scale resolution of modalities
        :param mask_src: (optional) Source mask.
        :param mask_dst: (optional) Destination mask.
        :param compatibility: (optional) Classes compatibility matrix, defaults to Potts model.
        :param custom_modality_downsamplers: A dictionary of modality downsampling functions.
        :param out_kernels_vis: Whether to return a tensor with kernels visualized with some step.
        :return: Loss function value.
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape

        device = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
        )

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        y_hat_unfolded = torch.abs(y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(N, C, 1, 1, height_pred, width_pred) - y_hat_unfolded)

        loss = torch.mean((kernels * y_hat_unfolded).view(N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred).sum(dim=2, keepdim=True))


        out = {
            'loss': loss.mean(),
        }

        if out_kernels_vis:
            out['kernels_vis'] = self._visualize_kernels(
                kernels, kernels_radius, height_input, width_input, height_pred, width_pred
            )

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = FeatureLoss._get_mesh(N, height_pred, width_pred, device)
                else:
                    assert modality in sample, \
                        f'Modality {modality} is listed in {i}-th kernel descriptor, but not present in the sample'
                    feature = sample[modality]
                    # feature = LocalSaliencyCoherence._downsample(
                    #     feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    # )
                feature /= sigma

                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * FeatureLoss._create_kernels_from_features(features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        assert features.dim() == 4, 'Features must be a NCHW batch'
        N, C, H, W = features.shape
        kernels = FeatureLoss._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius, radius, :, :].view(N, C, 1, 1, H, W)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        # kernels[:, :, radius, radius, :, :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, W, 1, dtype=torch.float32, device=device).view(1, 1, 1, W).repeat(N, 1, H, 1),
            torch.arange(0, H, 1, dtype=torch.float32, device=device).view(1, 1, H, 1).repeat(N, 1, 1, W)
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        assert img.dim() == 4, 'Unfolding requires NCHW batch'
        N, C, H, W = img.shape
        diameter = 2 * radius + 1
        return F.unfold(img, diameter, 1, radius).view(N, C, diameter, diameter, H, W)

    @staticmethod
    def _visualize_kernels(kernels, radius, height_input, width_input, height_pred, width_pred):
        diameter = 2 * radius + 1
        vis = kernels[:, :, :, :, radius::diameter, radius::diameter]
        vis_nh, vis_nw = vis.shape[-2:]
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw)
        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred-vis.shape[3], 0, height_pred-vis.shape[2]])
        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis



# 1. 确保 StructureGuidedALSC 继承自 FeatureLoss (关键修改!)
class StructureGuidedALSC(FeatureLoss): 
    def __init__(self):
        # 2. 初始化父类
        super(StructureGuidedALSC, self).__init__()
        
        # 定义拉普拉斯核 (二阶差分)，用于提取结构边界
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32)
        # 使用 register_buffer 确保它会被移动到 GPU，且不会被视为可训练参数
        self.register_buffer('laplacian_kernel', kernel.view(1, 1, 3, 3))

    def get_semantic_boundary(self, f1_feature):
        """
        从 Backbone 的浅层特征 f1_feature 提取语义边界
        输入: [B, C, H, W]
        输出: [B, 1, H, W] 的归一化边界图
        """
        # 1. 降维: 使用 Max Pooling 聚合通道信息
        f1_compressed, _ = torch.max(f1_feature, dim=1, keepdim=True) # [B, 1, H, W]
        
        # 2. 归一化: 确保特征在做差分前处于统一尺度 (Instance Norm)
        f1_norm = F.instance_norm(f1_compressed)
        
        # 3. 计算二阶差分 (Laplacian)
        # padding=1 保持尺寸不变
        boundary_response = F.conv2d(f1_norm, self.laplacian_kernel, padding=1)
        
        # 4. 取绝对值 (Magnitude) 并 Detach (阻断梯度)
        boundary_mag = torch.abs(boundary_response).detach()
        
        return boundary_mag

    def _normalize_map(self, feat_map):
        """将特征图归一化到 [0, 1] 区间，防止 sigma 失效"""
        B, C, H, W = feat_map.shape
        feat_flat = feat_map.view(B, -1)
        min_v = feat_flat.min(dim=1, keepdim=True)[0]
        max_v = feat_flat.max(dim=1, keepdim=True)[0]
        # 添加 1e-6 防止除以零
        norm_map = (feat_flat - min_v) / (max_v - min_v + 1e-6)
        return norm_map.view(B, C, H, W)

    def forward(self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input,
                structure_feature=None, custom_modality_downsamplers=None, out_kernels_vis=False):
        
        # 如果传入了 structure_feature (来自 backbone 的 F1), 则计算结构图并加入 sample
        if structure_feature is not None:
            # 1. 计算边界图
            struct_map = self.get_semantic_boundary(structure_feature)
            # 2. 上采样到与输入图像一致
            struct_map = F.interpolate(struct_map, size=(height_input, width_input), mode='bilinear', align_corners=False)
            # 3. 再次归一化，确保数值范围适配 sigma
            struct_map = self._normalize_map(struct_map)
            # 4. 加入 sample 字典
            sample['structure'] = struct_map

        # 3. 使用 super() 调用父类的 forward 方法 (关键修改!)
        return super(StructureGuidedALSC, self).forward(
            y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, 
            custom_modality_downsamplers=custom_modality_downsamplers, out_kernels_vis=out_kernels_vis
        )

