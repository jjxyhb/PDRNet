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

    out2, _, out3, out4, out5, out6, hook0 = net(image, )    #auc_out= (B,1,H,W) 

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
    loss2_lsc = loss_lsc(out2_, loss_lsc_kernels_desc_defaults, loss_lsc_radius, sample, image_.shape[2], image_.shape[3])['loss']


    loss2 =loss_ssc+ (criterion(out2, fg_label) + criterion(out2, bg_label)) + l * loss2_lsc + loss_intra[0]

    return loss2, loss2*0.0,loss2*0.0, loss2*0.0, loss2*0.0