import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
import math

from models.util import grad_mul_const
from models.loss import cal_nll_loss

class CCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']

        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)
        self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props*2)
 
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)

        self.max_video_len = 200

        # counterfactual
        self.fusion_mode = 'rubi'
        self.end_classif = True
        self.cross = False
        self.rectify = True
        self.weight = False
        self.gaussian_label = True
        self.beta_op, self.beta_on, self.beta_pn = 0.05, 0.15, 0.15
        # Q->A branch
        self.q_pos = copy.deepcopy(self.fc_comp)
        self.q_ref = copy.deepcopy(self.fc_comp)
        self.q_neg_1 = copy.deepcopy(self.fc_comp)
        self.q_neg_2 = copy.deepcopy(self.fc_comp)
        if self.end_classif:
            self.q_pos_additional = nn.Linear(self.vocab_size, self.vocab_size)
            self.q_ref_additional = nn.Linear(self.vocab_size, self.vocab_size)
            self.q_neg_1_additional = nn.Linear(self.vocab_size, self.vocab_size)
            self.q_neg_2_additional = nn.Linear(self.vocab_size, self.vocab_size)

        self.constant = nn.Parameter(torch.tensor(0.0))

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, mode='train', **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # generate Gaussian masks
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        gauss_param = torch.sigmoid(self.fc_gauss(h[:, -1])).view(bsz*self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # downsample for effeciency
        props_len = n_frames//4
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)
        
        # semantic completion
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)
        hq1, hq = words_feat1, words_feat
        q_out = self.counterfactual_module(words_logit, hq1, self.q_pos, self.q_pos_additional, mode='train')
        cf_loss_pos = self.counterfactual_loss(q_out, words_id1, words_mask1)   # (B*p)
        # min_pos_loss, min_idx = torch.min(cf_loss_pos.reshape(bsz, self.num_props), dim=-1)     # idx: (B)

        if self.use_negative:
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, gauss_center, gauss_width, kwargs['epoch'])
            
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
            q_neg_out_1 = self.counterfactual_module(neg_words_logit_1, hq1, self.q_neg_1, self.q_neg_1_additional, mode='train')
            cf_loss_neg_1 = self.counterfactual_loss(q_neg_out_1, words_id1, words_mask1)   # (B*p)
  
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)
            q_neg_out_2 = self.counterfactual_module(neg_words_logit_2, hq1, self.q_neg_2, self.q_neg_2_additional, mode='train')
            cf_loss_neg_2 = self.counterfactual_loss(q_neg_out_2, words_id1, words_mask1)   # (B*p)

            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
            q_ref_out = self.counterfactual_module(ref_words_logit, hq, self.q_ref, self.q_ref_additional, mode='train')
            cf_loss_ref = self.counterfactual_loss(q_ref_out, words_id, words_mask)   # (B)

            # neg_words_logit_1 = q_neg_out_1['logits_cfvqa']
            # neg_words_logit_2 = q_neg_out_2['logits_cfvqa']
            # ref_words_logit = q_ref_out['logits_cfvqa']

        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        cf_loss = 0 #cf_loss_pos + cf_loss_ref + cf_loss_neg_1 + cf_loss_neg_2

        if mode == 'test':

            words_logit = q_out['logits_cfvqa']

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'cf_loss': cf_loss,
            'cf_neg_loss_1': cf_loss_neg_1,
            'cf_neg_loss_2': cf_loss_neg_2,
            'cf_ref_loss': cf_loss_ref,
            'cf_pos_loss': cf_loss_pos,
        }

    def counterfactual_loss(self, prediction, gt, q_mask, sim=None):
        v_pre = prediction['logits_tar']
        # q_pre = prediction['logits_src']
        z_q = prediction['z_nde']
        z_qkv = prediction['logits_all']
        # KL loss
        p_te = torch.nn.functional.softmax(z_qkv, -1).clone().detach()
        p_nde = torch.nn.functional.softmax(z_q, -1)
        kl_loss = - p_te * p_nde.log()
        kl_loss = kl_loss.sum(1).mean()
        # prediction loss
        loss_v, _ = cal_nll_loss(v_pre, gt, q_mask)
        loss_z, _ = cal_nll_loss(z_qkv, gt, q_mask)
        cf_loss = kl_loss + loss_v + loss_z
        # cf_loss = torch.mean(kl_loss) + torch.mean(loss_v) + torch.mean(loss_z)
        return cf_loss

    def counterfactual_module(self, fusion, tar, tar_head, tar_add=None, mode='train'):
        out = {}

        # tar = grad_mul_const(tar, 0.0)  # don't backpropagate
        tar_pred = tar_head(tar).squeeze(-1)  # N * T * D -> N * T

        # src = grad_mul_const(src, 0.0)  # don't backpropagate
        # src_pred = src_head(src)  # N * D -> N * T

        # both q, k and v are the facts
        z_qkv = self.fusion(fusion=fusion, target=tar_pred,
                            fusion_fact=True, target_fact=True)  # te = total effect
        # q is the fact while k and v are the counterfactuals
        z_q = self.fusion(fusion=fusion, target=tar_pred,
                          fusion_fact=False, target_fact=True)  # nie = natural indirect effect

        logits_cfvqa = z_qkv - z_q

        if self.end_classif:
            tar_out = tar_add(tar_pred)  # N * T
            # src_out = src_add(src_pred)  # N * T -> N * T
        else:
            tar_out = tar_pred
            # src_out = src_pred

        if mode == 'train':
            out['logits_all'] = z_qkv  # for optimization
            # out['logits_vq'] = logits  # predictions of the original VQ branch, i.e., NIE
            out['logits_cfvqa'] = logits_cfvqa  # predictions of CFVQA, i.e., TIE
            out['logits_tar'] = tar_out  # for optimization
            # out['logits_src'] = src_out  # for optimization
            out['z_nde'] = self.fusion(fusion.clone().detach(), tar_pred.clone().detach(),
                                       fusion_fact=False, target_fact=True)  # z_q for kl optimization with no grad
            return out
        else:
            return logits_cfvqa

    def fusion(self, fusion, target, fusion_fact=False, target_fact=False):

        fusion, target = self.transform(fusion, target, fusion_fact=fusion_fact,
                                                target_fact=target_fact)

        if self.fusion_mode == 'rubi':
            z = fusion * torch.sigmoid(target)

        elif self.fusion_mode == 'hm':
            z = fusion * target
            z = torch.log(z + eps) - torch.log1p(z)

        elif self.fusion_mode == 'sum':
            z = fusion + target
            z = torch.log(torch.sigmoid(z) + eps)

        return z

    def transform(self, fusion, target, fusion_fact=False, target_fact=False):
        gpu_id = self.constant.device
        # cuda_str = 'cuda:{}'.format(gpu_id)
        device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
        # torch.cuda.set_device(gpu_id)
        if not fusion_fact:
            fusion = self.constant * torch.ones_like(fusion).to(device)

        if not target_fact:
            target = self.constant * torch.ones_like(target).to(device)

        if self.fusion_mode == 'hm':
            fusion = torch.sigmoid(fusion)
            target = torch.sigmoid(target)

        return fusion, target
    
    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]

    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
