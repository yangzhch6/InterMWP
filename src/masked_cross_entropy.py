import torch
from torch.nn import functional


def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def logic_cross_entropy(logic, inter, inter_mask, logic_op_mask):
    logic_flat = logic.view(-1, logic.size(-1)) # (B*max_output_len) x Output_size
    log_probs_flat = functional.log_softmax(logic_flat, dim=1)  
    logic_op_mask = logic_op_mask.view(-1, logic.size(-1)).float()   
    log_probs_flat = torch.where(logic_op_mask>0, log_probs_flat, -1e12*torch.ones_like(log_probs_flat))
    _, predicts = torch.max(log_probs_flat, dim=1)
    predicts = predicts.view(*inter.size())

    inter_flat = inter.view(-1, 1)
    zeros = torch.zeros_like(inter_flat)
    ones = torch.ones_like(inter_flat)

    inter_mask = inter_mask.view(-1, 1)

    # count the number of all logic node
    logic_num = inter_mask.sum()
    
    # logic[0] should be weighted  to 0.06, else will be weighted 1, 
    # logic[-1] is num nodes and paddings, should be weighted as 0
    # mask = torch.where(inter_flat == -1, zeros, ones)
    # loss_weight = torch.where(inter_flat == 0, 0.06*inter_mask, inter_mask.float())
    loss_weight = inter_mask

    # change inter's -1 to 0, avoid Index Select Error
    inter_flat = torch.where(inter_flat == -1, zeros, inter_flat)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=inter_flat)
    losses_flat = losses_flat * loss_weight
    losses = losses_flat.view(*inter.size())
    loss = losses.sum() / logic_num

    correct = (predicts == inter) * inter_mask.view(*inter.size())
    acc_num_fine = correct.sum()
    
    predicts_mask = predicts * inter_mask.view(*inter.size())
    logic_mask = inter * inter_mask.view(*inter.size())

    acc_num_coarse = 0
    for i in range(predicts_mask.shape[0]):
        if torch.equal(predicts_mask[i,:], logic_mask[i,:]):
            acc_num_coarse += 1

    return loss, predicts.cpu().numpy(), acc_num_coarse, acc_num_fine, logic_num


def masked_cross_entropy(logits, target, length):
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    logits_flat = logits.view(-1, logits.size(-1)) # (B*max_output_len) x Output_size
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)     

    _, predicts = torch.max(log_probs_flat, dim=1)
    predicts = predicts.view(*target.size())

    target_flat = target.view(-1, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()

    correct = (predicts == target) * mask
    accurate = (torch.sum(correct, dim=1) == torch.sum(mask, dim=1))
    accurate = torch.sum(accurate, dim=0)

    return loss, accurate


def masked_cross_entropy_without_logit(logits, target, length):
    if torch.cuda.is_available():
        length = torch.LongTensor(length).cuda()
    else:
        length = torch.LongTensor(length)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))

    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = torch.log(logits_flat + 1e-12)

    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())

    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    # if loss.item() > 10:
    #     print(losses, target)
    return loss

