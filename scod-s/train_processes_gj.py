import torch.nn.functional as F
import torch
from feature_loss import *
from tools import *
from utils import ramps
import numpy as np
device=torch.device("cuda:0")
criterion = torch.nn.CrossEntropyLoss(weight=None, ignore_index=255, reduction='mean').to(device)#.cuda()
loss_lsc = FeatureLoss().to(device)#.cuda()
loss_lsc_kernels_desc_defaults = [{"weight": 1, "xy": 6, "rgb": 0.1}]
loss_lsc_radius = 5
l = 0.3

# 初始化改进后的 Loss (在 loop 外)
loss_alsc_improved = StructureGuidedALSC().to(device)

def get_dynamic_sigma(current_epoch, max_epoch=120):
    """
    计算动态 Sigma: 随训练进行线性衰减
    初期 (0.3): 允许模糊边界，快速覆盖目标
    后期 (0.05): 强制精确边界，利用结构图切断背景
    """
    sigma_start = 0.3
    sigma_end = 0.05
    if current_epoch >= max_epoch:
        return sigma_end
    decay = (sigma_start - sigma_end) * (current_epoch / max_epoch)
    return sigma_start - decay

def train_loss(image, mask, net, ctx, ft_dct, w_ft=.1, ft_st=2, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):
    
    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo'] # total epochs
    
    # ... (原有的大部分增强和预处理逻辑保持不变) ...
    # ... (do_moretrsf, image_tr, loss_intra 计算等) ...

    # 1. 网络前向传播
    # 注意: 我们修改了 Net 的返回值，增加了 f1_feat (原代码中的 hook0 位置或新增位置)
    # out2, ..., out6, f1_feat = net(image) 
    # 假设你修改 Net 后返回的是 out0, 0, out0, out0, out0, out0, bk_stage2
    out2, _, out3, out4, out5, out6, f1_feat = net(image)

    # ... (out_proc 处理 sigmoid 和 cat 保持不变) ...
    out2, out3, out4, out5, out6 = out_proc(out2, out3, out4, out5, out6)
    
    # ... (loss_ssc 对比损失计算保持不变) ...
    loss_ssc = Contrastive_loss(out2_s, out2_scale.detach()) # 假设你有这个 context

    # ... (GT 处理保持不变) ...
    gt = mask.squeeze(1).long()
    bg_label = gt.clone()
    fg_label = gt.clone()
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255

    # 2. 准备 ALSC Loss 的输入
    image_down = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
    sample = {'rgb': image_down}
    
    # 下采样预测图以节省显存
    out2_down = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=False)

    # 3. 计算动态 Sigma
    curr_sigma = get_dynamic_sigma(epoch, max_epoch=t_epo)
    
    # 4. 构造 Kernel Descriptor
    # 加入 'structure' 项，并使用动态 sigma
    dynamic_kernels_desc = [{
        "weight": 1, 
        "xy": 6, 
        "rgb": 0.1, 
        "structure": curr_sigma  # <--- 动态调整的结构约束力度
    }]
    
    # 5. 计算改进后的 ALSC Loss
    # 传入 f1_feat (作为 structure_feature)
    loss2_lsc_dict = loss_alsc_improved(
        y_hat_softmax=out2_down, 
        kernels_desc=dynamic_kernels_desc, 
        kernels_radius=loss_lsc_radius, 
        sample=sample, 
        height_input=image_down.shape[2], 
        width_input=image_down.shape[3],
        structure_feature=f1_feat # <--- 传入 Deep Feature
    )
    loss2_lsc = loss2_lsc_dict['loss']




def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=150):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, consistency_rampup)

def get_transform(ops=[0,1,2]):
    '''One of flip, translate, crop'''
    op = np.random.choice(ops)
    if op==0:
        flip = np.random.randint(0, 2)
        pp = Flip(flip)
    elif op==1:
        # pp = Translate(0.3)
        pp = Translate(0.15)
    elif op==2:
        #pp = Crop(0.7, 0.7)
        pp = Crop(0.7, 0.7)
    return pp
    #pp=Translate(0.15)
    #return pp
def get_color_tranform(ops=[0,1,2,3,4,5]):
    op=np.random.choice(ops)
    if op==3:
        pp=GaussianBlur(5)
        return pp
    if op==4:
        pp=mask()
        return pp
    if op==5:
        pp=Color_jitter()
        return pp


def train_loss(image, mask, net, ctx, ft_dct, w_ft=.1, ft_st = 2, ft_fct=.5, ft_head=True, mtrsf_prob=1, ops=[0,1,2], w_l2g=0, l_me=0.1, me_st=50, me_all=False, multi_sc=0, l=0.3, sl=1):
    #print(mask.shape) #[16,1,320,320]+torch.tensor
    #image:tensor+[16,3,320,320]


    if ctx:
        epoch = ctx['epoch']
        global_step = ctx['global_step']
        sw = ctx['sw']
        t_epo = ctx['t_epo']


    ######  saliency structure consistency loss  ######

    do_moretrsf = np.random.uniform()<mtrsf_prob
    if do_moretrsf:
        pre_color_transform=get_color_tranform()
        if pre_color_transform ==None:
            pre_transform = get_transform(ops)
            image_tr=pre_transform(image)
            #image_tr=image
        else:
            #image_tr=image

            image_tr=pre_color_transform(image,mask)
        #image_tr = pre_transform(image)
        large_scale = True
    else:
        large_scale = np.random.uniform() < multi_sc
        image_tr = image
    sc_fct = 0.5 if large_scale else 0.3
    image_scale = F.interpolate(image_tr, scale_factor=sc_fct, mode='bilinear', align_corners=False)

    out2, _, out3, out4, out5, out6, f1_feat = net(image, )    #auc_out= (B,1,H,W) 

    out2_s, _, out3_s, out4_s, out5_s, out6_s, _ = net(image_scale, )
    ### Calc intra_consisten loss (l2 norm) / entorpy
    loss_intra = []
    #me_st too large >50 epoch
    if epoch>=me_st:
        def entrp(t):
            etp = -(F.softmax (t, dim=1) * F.log_softmax (t, dim=1)).sum(dim=1)
            msk = (etp<0.5)
            return (etp*msk).sum() / (msk.sum() or 1)
        me = lambda x: entrp(torch.cat((x*0, x), 1)) # orig: 1-x, x
        if not me_all:
            e = me(out2)
            loss_intra.append(e * get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st))
            loss_intra = loss_intra + [0,0,0,0]
            sw.add_scalar('intra entropy', e.item(), global_step)
        elif me_all:
            ga = get_current_consistency_weight(epoch-me_st, consistency=l_me, consistency_rampup=t_epo-me_st)
            for i in [out2, out3, out4, out5, out6]:
                loss_intra.append(me(i)*ga)
            sw.add_scalar('intra entropy', loss_intra[0].item(), global_step)
    else:
        loss_intra.extend([0 for _ in range(5)])

    # def out_proc(out2, out3, out4, out5, out6,fg,bg):
    #     a = [out2, out3, out4, out5, out6,fg,bg]

    def out_proc(out2, out3, out4, out5, out6):
        a = [out2, out3, out4, out5, out6]
        a = [i.sigmoid() for i in a]
        a = [torch.cat((1 - i, i), 1) for i in a]
        return a
    #sigmoid
    out2, out3, out4, out5, out6 = out_proc(out2, out3, out4, out5, out6) #init

    #out2, out3, out4, out5, out6,fg,bg = out_proc(out2, out3, out4, out5, out6,fg,bg)
    # the size of out_s is be transformered
    #out2_s, out3_s, out4_s, out5_s, out6_s,fg_s,bg_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s,fg_s,bg_s)
    out2_s, out3_s, out4_s, out5_s, out6_s = out_proc(out2_s, out3_s, out4_s, out5_s, out6_s)
    if not do_moretrsf:
        out2_scale = F.interpolate(out2[:, 1:2], scale_factor=sc_fct, mode='bilinear', align_corners=False)
        out2_s = out2_s[:, 1:2]
        # out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=0.3/sc_fct, mode='bilinear', align_corners=False)
    else:
        #out2_ss=out2
        if pre_color_transform==None:
            out2_ss = pre_transform(out2)
            #out2_ss=out2
        else:
            out2_ss=out2
        out2_scale = F.interpolate(out2_ss[:, 1:2], scale_factor=1.0, mode='bilinear', align_corners=False)
        out2_s = F.interpolate(out2_s[:, 1:2], scale_factor=1.0/sc_fct, mode='bilinear', align_corners=False)
    # wen ding xing f
    loss_ssc = Contrastive_loss(out2_s, out2_scale.detach())

    gt = mask.squeeze(1).long() ##"0" stands for backgrounds, "1" for foregrounds, and "255" for unlabeled regions.
    #mask has  
    bg_label = gt.clone()
    fg_label = gt.clone()
    
    bg_label[gt != 0] = 255
    fg_label[gt == 0] = 255

    image_ = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
    sample = {'rgb': image_}
    # print('sample :', image_.max(), image_.min(), image_.std())
    #out2_ = F.interpolate(out2[:, 0:1], scale_factor=0.25, mode='bilinear', align_corners=False)
    out2_ = F.interpolate(out2[:, 1:2], scale_factor=0.25, mode='bilinear', align_corners=False)
    
    curr_sigma = get_dynamic_sigma(epoch, max_epoch=t_epo)
    
    # 4. 构造 Kernel Descriptor
    # 加入 'structure' 项，并使用动态 sigma
    dynamic_kernels_desc = [{
        "weight": 1, 
        "xy": 6, 
        "rgb": 0.1, 
        "structure": curr_sigma  # <--- 动态调整的结构约束力度
    }]
    
    # 5. 计算改进后的 ALSC Loss
    # 传入 f1_feat (作为 structure_feature)
    loss2_lsc_dict = loss_alsc_improved(
        y_hat_softmax=out2_, 
        kernels_desc=dynamic_kernels_desc, 
        kernels_radius=loss_lsc_radius, 
        sample=sample, 
        height_input=image_.shape[2], 
        width_input=image_.shape[3],
        structure_feature=f1_feat # <--- 传入 Deep Feature
    )
    loss2_lsc = loss2_lsc_dict['loss']


    loss2 =loss_ssc+ (criterion(out2, fg_label) + criterion(out2, bg_label)) + l * loss2_lsc + loss_intra[0]

    return loss2, loss2*0.0,loss2*0.0, loss2*0.0, loss2*0.0