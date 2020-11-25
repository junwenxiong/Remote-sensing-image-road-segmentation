import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class SegmentationLosses(object):
    def __init__(self,
                 weight=None,
                 batch_average=True,
                 ignore_index=255,
                 cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """"""
        if mode == 'ce':
            return self.cross_entropy_loss
        if mode == 'focal':
            return self.focal_loss
        if mode == 'lovasz':
            return self.lovasz
        if mode == 'dice_bce':
            return self.dice_bce
        if mode == 'dice':
            return self.dice
        if mode == 'mixed':
            return self.mixedloss

    def dice(self, logit, target):
        if self.cuda:
            criterion = DiceLoss().cuda()

        else:
            criterion = DiceLoss()

        loss = criterion(logit, target)
        return loss

    def lovasz(self, logit, target):
        if self.cuda:
            criterion = LovaszSoftmax().cuda()
        else:
            criterion = LovaszSoftmax()
        loss = criterion(logit, target)
        return loss

    def cross_entropy_loss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean')
        if self.cuda:
            criterion = criterion.cuda()

        # target should be long type because the crossentropy function
        # In the formula, target is used to index the output logit for
        # the current target class(note the indexing in x[class])
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        loss = criterion(logit, target)

        if self.batch_average:
            loss /= n

        return loss

    def focal_loss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean')

        if self.cuda:
            criterion = criterion.cuda()
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt)**gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def dice_bce(self, logit, target):
        if self.cuda:
            criterion = dice_bce_loss().cuda()
        else:
            criterion = dice_bce_loss()
        loss = criterion(logit, target)
        return loss

    def mixedloss(self, logit, target):
        if self.cuda:
            criterion = MixedLoss(alpha=10.0, gamma=2.0).cuda()
        else:
            criterion = MixedLoss(alpha=10.0, gamma=2.0)
        loss = criterion(logit, target)
        return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(
                torch.dot(
                    loss_c_sorted,
                    torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        # print(inputs.shape, targets.shape) # (batch size, class_num, x,y,z), (batch size, 1, x,y,z)
        inputs, targets = self.prob_flatten(inputs, targets)
        # print(inputs.shape, targets.shape)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses

    def __call__(self, inputs, targets):
        loss = self.forward(inputs, targets)
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        # assert pred.shape == target.shape

        N = target.size(0)
        smooth = 1

        input_flat = pred.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) +
                    smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)

        # loss = 1 - loss.sum() / N

        return loss

    # def __call__(self, pred, targets):
    #     loss = self.forward(pred, targets)
    #     return loss


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        #self.bce_loss = nn.BCELoss()
        weight = torch.tensor([0.1])
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=weight)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        y_true1 = y_true.float()
        y_pred1 = y_pred.float()

        a = self.bce_loss(y_pred1, y_true1)

        b = self.soft_dice_loss(y_true1, y_pred1)
        return a + b


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                'Target size ({}) must be the same as the input size ({})'.
                format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + (
            (-max_val).exp() + (-input - max_val).exp()).long()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, target, pred):
        # import pdb
        # pdb.set_trace()
        loss = self.alpha * self.focal(pred, target) - torch.log(
            self.dice(pred, target))
        return loss.mean()


def flatten(tensor):

    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose : (N, C, D, H, W) -> （C, N, D, H, W）
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten : (C, N, D, H ,W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice for imbalance classes
        https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py
        """

        super(GDiceLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, pred, target):
        shp_x = pred.shape  # (batch_size, class_num, x, y, z )
        shp_y = target.shape  #

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                target = target.view(shp_y[0], 1, *shp_y[1:])

            if all([i == i for i, j in zip(pred.shape, target.shape)]):
                y_onehot = target
            else:
                target = target.long()
                y_onehot = torch.zeros(shp_x)
                if pred.device.type == 'cuda':
                    y_onehot = y_onehot.cuda(pred.device.index)
                y_onehot.scatter_(1, target, 1)

        if self.apply_nonlin is not None:
            pred = self.apply_nonlin(pred)

        input = flatten(pred)
        output = flatten(y_onehot)
        output = output.float()
        output_sum = output.sum(-1)
        class_weights = Variable(
            1. / (output_sum * output_sum).clamp(min=self.smooth),
            requires_grad=False)

        intersect = (input * output).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + output).sum(-1) * class_weights).sum()

        return -2. * intersect / denominator.clamp(min=self.smooth)


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=False)
    loss2 = DiceLoss()
    a = torch.rand(1, 3, 7, 7)
    b = torch.rand(1, 7, 7)
    # print(loss.cross_entropy_loss(a,b).item())
    # print(loss.focal_loss(a, b, gamma=0, alpha=None).item())
    # print(loss.focal_loss(a, b, gamma=2, alpha=0.5).item())
    print(loss2(a, b).item())