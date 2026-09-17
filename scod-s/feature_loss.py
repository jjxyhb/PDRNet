import torch
import torch.nn.functional as F

"""
The following code is modified from the file in https://github.com/siyueyu/SCWSSOD/blob/f8650567cbbc8df5bf6edc32a633c47a885574cd/lscloss.py.
Credit for them.

Modified to support Confidence-Guided Adaptive Multi-mask Fusion (CGAMF) 
for Weakly-Supervised Camouflaged Object Detection.
"""
class FeatureLoss(torch.nn.Module):
    """
    Modified from Gated CRF loss to support:
    1) deep feature affinity
    2) confidence-interval truncation with tri-state consensus mask

    tri-state consensus:
        1   -> reliable foreground
        0   -> reliable background
        255 -> uncertain
    """

    def forward(
        self,
        y_hat_softmax,
        kernels_desc,
        kernels_radius,
        sample,
        height_input,
        width_input,
        mask_src=None,
        mask_dst=None,
        compatibility=None,
        custom_modality_downsamplers=None,
        out_kernels_vis=False,
        mask_consensus=None,
    ):
        """
        y_hat_softmax: [N, C, H, W]
        sample: dict containing feature modalities, e.g. {'deep_f': ...}
        mask_consensus: [N, 1, H, W], values in {0, 1, 255}
        """
        assert y_hat_softmax.dim() == 4, 'Prediction must be a NCHW batch'
        N, C, height_pred, width_pred = y_hat_softmax.shape
        dev = y_hat_softmax.device

        assert width_input % width_pred == 0 and height_input % height_pred == 0 and \
               width_input * height_pred == height_input * width_pred, \
            f'[{width_input}x{height_input}] !~= [{width_pred}x{height_pred}]'

        # Create base affinity kernels using xy + deep_f
        kernels = self._create_kernels(
            kernels_desc,
            kernels_radius,
            sample,
            N,
            height_pred,
            width_pred,
            dev,
            custom_modality_downsamplers
        )

        # =========================================================
        # Confidence-interval truncation (tri-state)
        # Only allow propagation when:
        #   - center is reliable
        #   - neighbor is reliable
        #   - center and neighbor belong to same reliable class
        # =========================================================
        if mask_consensus is not None:
            assert mask_consensus.dim() == 4, 'mask_consensus must be a NCHW batch'

            consensus_unfolded = self._unfold(mask_consensus.float(), kernels_radius)
            consensus_center = consensus_unfolded[:, :, kernels_radius, kernels_radius, :, :] \
                .view(N, 1, 1, 1, height_pred, width_pred)

            valid_center = (consensus_center != 255).float()
            valid_neigh = (consensus_unfolded != 255).float()
            same_region = (consensus_center == consensus_unfolded).float()

            indicator = valid_center * valid_neigh * same_region
            kernels = kernels * indicator

        # Unfold prediction
        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)

        # |p_i - p_j|
        y_hat_unfolded = torch.abs(
            y_hat_unfolded[:, :, kernels_radius, kernels_radius, :, :].view(
                N, C, 1, 1, height_pred, width_pred
            ) - y_hat_unfolded
        )

        loss = torch.mean(
            (
                kernels * y_hat_unfolded
            ).view(
                N, C, (kernels_radius * 2 + 1) ** 2, height_pred, width_pred
            ).sum(dim=2, keepdim=True)
        )

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
        kernels_desc,
        kernels_radius,
        sample,
        N,
        height_pred,
        width_pred,
        device,
        custom_modality_downsamplers
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

                    # 如果输入尺寸不匹配，则自适应下采样到预测尺寸
                    if feature.shape[2] != height_pred or feature.shape[3] != width_pred:
                        feature = FeatureLoss._downsample(
                            feature, modality, height_pred, width_pred, custom_modality_downsamplers
                        )

                feature = feature / sigma
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
        vis = vis.permute(0, 1, 4, 2, 5, 3).contiguous().view(
            kernels.shape[0], 1, diameter * vis_nh, diameter * vis_nw
        )

        if vis.shape[2] > height_pred:
            vis = vis[:, :, :height_pred, :]
        if vis.shape[3] > width_pred:
            vis = vis[:, :, :, :width_pred]
        if vis.shape[2:] != (height_pred, width_pred):
            vis = F.pad(vis, [0, width_pred - vis.shape[3], 0, height_pred - vis.shape[2]])

        vis = F.interpolate(vis, (height_input, width_input), mode='nearest')
        return vis