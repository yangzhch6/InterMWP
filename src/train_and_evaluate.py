from src.masked_cross_entropy import *
from src.pre_data import *
from src.expressions_transfer import *
from src.models import *
import math
import copy
import torch
import torch.optim
import torch.nn.functional as f
import time

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120
USE_CUDA = torch.cuda.is_available()


class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)

def generate_tree_input(target, num_start):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    target_input_class = [[0,0,0,0,0]]*len(target)
    for i in range(len(target)):
        if target_input[i] >= num_start:
            target_input[i] = num_start
        target_input_class[i][target_input[i]] = 1 
    return torch.LongTensor(target_input), torch.FloatTensor(target_input_class)

def check_logic(predict, logic):
    acc_num = 0
    logic_num = 0
    predict = [d.cpu().item() for d in predict]
    for p, l  in zip(predict, logic):
        if l == -1:
            continue
        
        if p == l:
            acc_num += 1
        logic_num += 1
    return acc_num, logic_num


def compute_op_result(test_res, test_tar, output_lang, num_list):
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list)
    op_count = 0
    op_acc = 0
    if len(tar) == len(test):
        for i in range(len(tar)):
            if tar[i] in ['+', '-', '*', '/']:
                op_count += 1
                if tar[i] == test[i]:
                    op_acc += 1
    return op_count, op_acc

def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list):
    # print(test_res, test_tar)

    if test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list)
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar

def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, max_num_size, hidden_size):
    indices = list()
    masked_index = list()
    sen_len = encoder_outputs.size(0)
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), max_num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), max_num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, max_num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() # B x S x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # B x S x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, max_num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)

def get_all_number_encoder_outputs_ddp(encoder_outputs, num_pos, batch_size, num_size, hidden_size, local_rank):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.BoolTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.to(local_rank)
        masked_index = masked_index.to(local_rank)
    all_outputs = encoder_outputs.transpose(0, 1).contiguous() # B x S x H
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # B x S x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r

class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out, goal_stack=[]):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)
        self.goal_stack = copy.deepcopy(goal_stack)

class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal
        
def train_tree_cls(output, output_len, num_size, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
               predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_idx, 
               token_ids, token_type_ids, attention_mask):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    target = torch.LongTensor(output).transpose(0, 1)

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)

    encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_predict = []

    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_idx, batch_size, max(num_size),
                                                              encoder.config.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_output_len):
        num_score, op_score, goal, context, all_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        ## score
        predict_score = torch.cat((op_score, num_score), 1) # B x Output_size
        all_node_predict.append(predict_score) # [B x Output_size]

        # op's label of each node, nums node will be masked to 0
        node_op_label, _ = generate_tree_input(target[t].tolist(), num_start)
        if USE_CUDA:
            node_op_label = node_op_label.cuda()
        left_child, right_child, node_op_embedding = generate(goal, node_op_label, context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_op_embedding[idx].unsqueeze(0), False))
            else:
                current_num = all_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_predict = torch.stack(all_node_predict, dim=1)  # B x max_output_len x Output_size

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        output_len = torch.LongTensor(output_len).cuda()
        all_node_predict = all_node_predict.cuda()
        target = target.cuda()

    loss, accurate = masked_cross_entropy(all_node_predict, target, output_len)
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    return loss.item(), accurate.item()

def train_retriever(
        inter_multi_label, encoder, retriever, encoder_optimizer, encoder_scheduler, 
        retriever_optimizer, token_ids, token_type_ids, attention_mask,
        logic_token_ids, logic_attention_mask, logic_token_type_ids, 
    ):
    encoder.train()
    retriever.train()

    if USE_CUDA:
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        inter_multi_label = torch.tensor(inter_multi_label, dtype=torch.long).cuda()

        logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
        logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
        logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    retriever_optimizer.zero_grad()

    last_hidden_state, _ = encoder(token_ids, token_type_ids, attention_mask)
    logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
    output_embedding = torch.mean(last_hidden_state, dim=1)
    logic_embedding = torch.mean(logic_embedding_bert, dim=1)
    logic_score = retriever(output_embedding, logic_embedding)
    # print(logic_score)
    loss_function = multilabel_categorical_crossentropy
    loss = loss_function(logic_score, inter_multi_label)

    loss.backward()
    # 累计梯度
    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()
    retriever_optimizer.step()

    return loss.item()

def evaluate_retriever(
        inter_multi_label, encoder, retriever, 
        token_ids, token_type_ids, attention_mask,
        logic_token_ids, logic_attention_mask, logic_token_type_ids,
    ):
    encoder.eval()
    retriever.eval()

    if USE_CUDA:
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        inter_multi_label = torch.tensor(inter_multi_label, dtype=torch.long).cuda()

        logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
        logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
        logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()

    last_hidden_state, _ = encoder(token_ids, token_type_ids, attention_mask)
    logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
    output_embedding = torch.mean(last_hidden_state, dim=1)
    logic_embedding = torch.mean(logic_embedding_bert, dim=1)
    logic_score = retriever(output_embedding, logic_embedding)
    loss_function = multilabel_categorical_crossentropy
    loss = loss_function(logic_score, inter_multi_label)
    return loss.item(), logic_score.tolist()

def retriever_output(
        encoder, retriever, 
        token_ids, token_type_ids, attention_mask,
        logic_token_ids, logic_attention_mask, logic_token_type_ids,
    ):
    encoder.eval()
    retriever.eval()

    if USE_CUDA:
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

        logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
        logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
        logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()

    last_hidden_state, _ = encoder(token_ids, token_type_ids, attention_mask)
    logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
    output_embedding = torch.mean(last_hidden_state, dim=1)
    logic_embedding = torch.mean(logic_embedding_bert, dim=1)
    logic_score = retriever(output_embedding, logic_embedding)
    return logic_score.tolist()


def recall_topk_line(predict_topk, target):
    acc = 0
    logic_set = list()
    for id in predict_topk:
        if id+1 in target:
            acc += 1
    for t in target:
        if t != 0 and t not in logic_set:
            logic_set.append(t)
    return acc, len(logic_set)

def recall_topk(predict_score, inter_prefix, k):
    predict_score = torch.tensor(predict_score)
    value, predict = predict_score.topk(k)
    predict = predict.tolist()

    acc_all = 0
    logic_count_all = 0
    for p_line, prefix in zip(predict, inter_prefix):
        acc, logic_count = recall_topk_line(p_line, prefix)
        acc_all += acc
        logic_count_all += logic_count
    global_recall = acc_all/logic_count_all
    precision = acc_all / (k*len(predict))
    return global_recall, precision, predict

def recall_topk_all(predict_score, inter_prefix):
    R_list = dict()
    P_list = dict()
    Predict_list = dict()
    for k in range(1, 7):
        recall, precision, predict = recall_topk(predict_score, inter_prefix, k)
        R_list[str(k)] = recall
        P_list[str(k)] = precision
        Predict_list[str(k)] = predict
    return R_list, P_list, Predict_list

def predict_line(line):
    ans = list()
    for idx in range(len(line)):
        if line[idx] == 1:
            ans.append(idx)
    return ans

def multilabel_categorical_crossentropy(logic_score, inter_multi_label):
    logic_score = (1- 2*inter_multi_label) * logic_score
    logic_score_neg = logic_score - inter_multi_label * 1e12
    logic_score_pos = logic_score - (1 - inter_multi_label) * 1e12
    zeros = torch.zeros_like(logic_score[:, 1]).unsqueeze(1)
    logic_score_neg = torch.cat((logic_score_neg, zeros), dim=1)
    logic_score_pos = torch.cat((logic_score_pos, zeros), dim=1)
    neg_loss = torch.logsumexp(logic_score_neg, dim=-1)
    pos_loss = torch.logsumexp(logic_score_pos, dim=-1)
    loss = torch.sum(neg_loss + pos_loss)
    return loss

def evaluate_tree_cls(generate_nums, encoder, predict, generate, merge, output_lang, 
                  num_pos, token_ids, token_type_ids, attention_mask, input_len_max, 
                  beam_size=5, max_length=MAX_OUTPUT_LENGTH, return_goal=False):

    # seq_mask = torch.ByteTensor(attention_mask)
    seq_mask = torch.BoolTensor(1, input_len_max).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)

    encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)
                current_goal_stack = copy.deepcopy(b.goal_stack)

                out_token = int(ti)
                current_out.append(out_token)
                current_goal_stack.append(current_embeddings.squeeze().cpu().detach().numpy())

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out, current_goal_stack))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break
    
    if return_goal:
        return beams[0].out, beams[0].goal_stack
    else:
        return beams[0].out


def train_logic_generator(
    encoder, logic_generator, 
    token_ids, token_type_ids, attention_mask,
    encoder_optimizer, logic_optimizer, encoder_scheduler,
    goal, logic_token_ids, logic_attention_mask, logic_token_type_ids,
    inter_prefix, inter_mask, logic_op_mask):
    
    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))

    encoder.train()
    logic_generator.train()

    encoder_optimizer.zero_grad()
    logic_optimizer.zero_grad()
    
    if USE_CUDA:
        seq_mask = seq_mask.cuda()
        goal = goal.cuda()
        logic_op_mask = torch.tensor(logic_op_mask, dtype=torch.long).cuda()
        inter_prefix = torch.tensor(inter_prefix, dtype=torch.long).cuda()
        inter_mask = torch.tensor(inter_mask, dtype=torch.long).cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
        logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
        logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()
    
    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()

    logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
    logic_embedding = torch.mean(logic_embedding_bert, dim=1)

    score = logic_generator(goal, encoder_outputs, seq_mask, logic_embedding)
    loss, predicts, acc_num_coarse, acc_num_fine, logic_num = \
        logic_cross_entropy(score, inter_prefix, inter_mask, logic_op_mask)
    loss.backward()

    encoder_optimizer.step()
    encoder_scheduler.step()    
    logic_optimizer.step()

    return loss.item(), score, acc_num_coarse

def evaluate_logic_generator(
    encoder, logic_generator, 
    token_ids, token_type_ids, attention_mask,
    goal, logic_token_ids, logic_attention_mask, logic_token_type_ids,
    inter_prefix, inter_mask, logic_op_mask):
    
    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))

    encoder.eval()
    logic_generator.eval()
    with torch.no_grad():
        if USE_CUDA:
            seq_mask = seq_mask.cuda()
            goal = goal.cuda()
            logic_op_mask = torch.tensor(logic_op_mask, dtype=torch.long).cuda()
            inter_prefix = torch.tensor(inter_prefix, dtype=torch.long).cuda()
            inter_mask = torch.tensor(inter_mask, dtype=torch.long).cuda()
            token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
            logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
            logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
            logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()
        
        encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
        encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()

        logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
        logic_embedding = torch.mean(logic_embedding_bert, dim=1)

        score = logic_generator(goal, encoder_outputs, seq_mask, logic_embedding)
        loss, predicts, acc_num_coarse, acc_num_fine, logic_num = \
            logic_cross_entropy(score, inter_prefix, inter_mask, logic_op_mask)

    return loss.item(), score, acc_num_coarse

def test_logic_generator( # line
    encoder, logic_generator, 
    token_ids, token_type_ids, attention_mask,
    goal, logic_token_ids, logic_attention_mask, logic_token_type_ids,
    output_full, inter_full, predict_output, logic_op_mask):
    
    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))

    encoder.eval()
    logic_generator.eval()
    with torch.no_grad():
        if USE_CUDA:
            seq_mask = seq_mask.cuda()
            goal = goal.cuda()
            logic_op_mask = torch.tensor(logic_op_mask, dtype=torch.long).cuda()
            token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
            attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
            logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
            logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
            logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()
        
        encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
        encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()

        logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
        logic_embedding = torch.mean(logic_embedding_bert, dim=1)

        score = logic_generator(goal, encoder_outputs, seq_mask, logic_embedding)

        score_flat = score.view(-1, score.size(-1)) # (B*max_output_len) x Output_size
        log_probs_flat = functional.log_softmax(score_flat, dim=1)   
        log_probs_flat = torch.where(logic_op_mask>0, log_probs_flat, -1e12*torch.ones_like(log_probs_flat))
        _, predicts = torch.max(log_probs_flat, dim=1)
        predicts = predicts.cpu().numpy()

        if predict_output in output_full:
            idx = output_full.index(predict_output)
            inter_prefix = inter_full[idx]
            formula_acc = 1
            logic_acc = 1
            for i in range(len(predicts)):
                if predict_output[i] < 4 and predicts[i] != inter_prefix[i]:
                    logic_acc = 0
                    break
            return formula_acc, logic_acc
        else:
            return 0, 0
        
def train_tree_logic(
    output, output_len, num_size, generate_nums,
    encoder, predict, generate, merge, 
    logic_predict, logic_generate, logic_merge, 
    encoder_optimizer, encoder_scheduler, 
    decoder_optimizer, output_lang, num_idx, 
    token_ids, token_type_ids, attention_mask,
    logic_token_ids, logic_attention_mask, 
    logic_token_type_ids, inter_prefix):

    seq_mask = torch.BoolTensor(attention_mask)
    seq_mask = (seq_mask == torch.BoolTensor(torch.zeros_like(seq_mask)))
    num_mask = []
    max_num_size = max(num_size) + len(generate_nums)
    for i in num_size:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.BoolTensor(num_mask)

    unk = output_lang.word2index["[UNK]"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    target = torch.LongTensor(output).transpose(0, 1)
    inter = torch.LongTensor(inter_prefix).transpose(0, 1)

    # [ [0.0]*predict.hidden_size ]
    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(token_ids)

    encoder.train()
    predict.train()
    generate.train()
    merge.train()
    logic_predict.train()
    logic_generate.train()
    logic_merge.train()

    if USE_CUDA:
        # print("convert tensor to cuda")
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
        logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
        logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
    logic_embedding = torch.mean(logic_embedding_bert, dim=1)

    encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    logic_node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_output_len = max(output_len)

    all_node_predict = []
    all_logic_predict = []

    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_idx, batch_size, max(num_size),
                                                              encoder.config.hidden_size)
    
    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    logic_embeddings_stacks = [[] for _ in range(batch_size)]

    left_childs = [None for _ in range(batch_size)]
    logic_left_childs = [None for _ in range(batch_size)]
    
    for t in range(max_output_len):
        num_score, op_score, goal, context, all_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, 
            padding_hidden, seq_mask, num_mask)
        
        logic_score, logic_goal, logic_context = logic_predict(
            logic_node_stacks, logic_left_childs, encoder_outputs, 
            padding_hidden, seq_mask, logic_embedding)

        ## score
        predict_score = torch.cat((op_score, num_score), 1) # B x Output_size
        all_node_predict.append(predict_score) # [B x Output_size]
        all_logic_predict.append(logic_score)

        # op's label of each node, nums node will be masked to 0
        node_op_label, _ = generate_tree_input(target[t].tolist(), num_start)
        if USE_CUDA:
            node_op_label = node_op_label.cuda()
        left_child, right_child, node_op_embedding = generate(goal, node_op_label, context)
        logic_left_child, logic_right_child, logic_op_embedding = logic_generate(
            goal+logic_goal, node_op_label, logic_context)
        
        left_childs = []
        logic_left_childs = []
        for idx, left, right, logic_left, logic_right, node_stack, \
            logic_stack, target_id, logic_id, exp_embed_stack, logic_embed_stack in  \
                zip(range(batch_size), left_child.split(1), right_child.split(1),
                logic_left_child.split(1), logic_right_child.split(1),
                node_stacks, logic_node_stacks, target[t].tolist(), inter[t].tolist(),
                embeddings_stacks, logic_embeddings_stacks):

            if len(node_stack) != 0:
                node = node_stack.pop()
                logic_node = logic_stack.pop()
            else:
                left_childs.append(None)
                logic_left_childs.append(None)
                continue

            if target_id < num_start:
                node_stack.append(TreeNode(right))
                node_stack.append(TreeNode(left, left_flag=True))
                exp_embed_stack.append(TreeEmbedding(node_op_embedding[idx].unsqueeze(0), False))

                logic_stack.append(TreeNode(logic_right))
                logic_stack.append(TreeNode(logic_left, left_flag=True))
                logic_embed_stack.append(TreeEmbedding(logic_op_embedding[idx].unsqueeze(0), False))
            else:
                current_num = all_nums_embeddings[idx, target_id - num_start].unsqueeze(0)
                current_logic = logic_embedding[logic_id].unsqueeze(0)
                while len(exp_embed_stack) > 0 and exp_embed_stack[-1].terminal:
                    sub_stree = exp_embed_stack.pop()
                    op = exp_embed_stack.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    logic_sub_stree = logic_embed_stack.pop()
                    logic_op = logic_embed_stack.pop()
                    current_logic = logic_merge(logic_op.embedding, logic_sub_stree.embedding, current_logic)
                exp_embed_stack.append(TreeEmbedding(current_num, True))
                logic_embed_stack.append(TreeEmbedding(current_logic, True))

            if len(exp_embed_stack) > 0 and exp_embed_stack[-1].terminal:
                left_childs.append(exp_embed_stack[-1].embedding)
                logic_left_childs.append(logic_embed_stack[-1].embedding)
            else:
                left_childs.append(None)
                logic_left_childs.append(None)

    all_node_predict = torch.stack(all_node_predict, dim=1)  # B x max_output_len x Output_size
    all_logic_predict = torch.stack(all_logic_predict, dim=1)

    target = target.transpose(0, 1).contiguous()
    inter = inter.transpose(0, 1).contiguous()

    if USE_CUDA:
        output_len = torch.LongTensor(output_len).cuda()
        all_node_predict = all_node_predict.cuda()
        target = target.cuda()
        inter = inter.cuda()

    mwp_loss, accurate = masked_cross_entropy(all_node_predict, target, output_len)
    logic_loss, accurate = masked_cross_entropy(all_logic_predict, inter, output_len)
    loss = mwp_loss + logic_loss
    loss.backward()

    # Update parameters with optimizers
    encoder_optimizer.step()
    encoder_scheduler.step()    
    decoder_optimizer.step()

    return loss.item(), mwp_loss.item(), logic_loss.item(), accurate.item()

def evaluate_tree_logic(
        generate_nums, encoder, predict, generate, merge, 
        logic_predict, logic_generate, logic_merge, output_lang, 
        num_pos, token_ids, token_type_ids, attention_mask, 
        logic_token_ids, logic_attention_mask, logic_token_type_ids, 
        input_len_max, beam_size=5, max_length=MAX_OUTPUT_LENGTH):

    # seq_mask = torch.ByteTensor(attention_mask)
    seq_mask = torch.BoolTensor(1, input_len_max).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.BoolTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()
    logic_predict.eval()
    logic_generate.eval()
    logic_merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

        token_ids = torch.tensor(token_ids, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        logic_token_ids = torch.tensor(logic_token_ids, dtype=torch.long).cuda()
        logic_token_type_ids = torch.tensor(logic_token_type_ids, dtype=torch.long).cuda()
        logic_attention_mask = torch.tensor(logic_attention_mask, dtype=torch.long).cuda()

    # Run words through encoder
    encoder_outputs, problem_output = encoder(token_ids, token_type_ids, attention_mask)
    logic_embedding_bert, _ = encoder(logic_token_ids, logic_token_type_ids, logic_attention_mask)
    logic_embedding = torch.mean(logic_embedding_bert, dim=1)

    encoder_outputs = encoder_outputs.transpose(0, 1).contiguous()

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]
    logic_node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.config.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    logic_embeddings_stacks = [[] for _ in range(batch_size)]

    left_childs = [None for _ in range(batch_size)]
    logic_left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]
    logic_beams = [TreeBeam(0.0, logic_node_stacks, logic_embeddings_stacks, logic_left_childs, [])]

    for t in range(max_length):
        current_beams = []
        current_logic_beams = []
        while len(beams) > 0:
            b = beams.pop()
            logic_b = logic_beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                current_logic_beams.append(logic_b)
                continue

            left_childs = b.left_childs
            logic_left_childs = logic_b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            logic_score, logic_goal, logic_context = logic_predict(
                logic_b.node_stack, logic_left_childs, encoder_outputs, 
                padding_hidden, seq_mask, logic_embedding)

            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            logic_score = nn.functional.log_softmax(logic_score, dim=1)

            topv, topi = out_score.topk(beam_size)
            logic_topv, logic_topi = logic_score.topk(beam_size)

            for tv, ti, logic_tv, logic_ti in zip(topv.split(1, dim=1), topi.split(1, dim=1), 
                              logic_topv.split(1, dim=1), logic_topi.split(1, dim=1)):
                
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                current_logic_stack = copy_list(logic_b.node_stack)
                current_logic_left_childs = []
                current_logic_embeddings_stacks = copy_list(logic_b.embedding_stack)
                current_logic_out = copy.deepcopy(logic_b.out)

                out_token = int(ti)
                logic_out_token = int(logic_ti)

                current_out.append(out_token)
                current_logic_out.append(logic_out_token)

                node = current_node_stack[0].pop()
                logic_node = current_logic_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
                    logic_left_child, logic_right_child, logic_label = logic_generate(
                        current_embeddings+logic_goal, generate_input, logic_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                    current_logic_stack[0].append(TreeNode(logic_right_child))
                    current_logic_stack[0].append(TreeNode(logic_left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                    current_logic_embeddings_stacks[0].append(TreeEmbedding(logic_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)
                    current_logic = logic_embedding[logic_out_token].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)

                        logic_sub_stree = current_logic_embeddings_stacks[0].pop()
                        logic_op = current_logic_embeddings_stacks[0].pop()
                        current_logic = logic_merge(logic_op.embedding, logic_sub_stree.embedding, current_logic)

                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                    current_logic_embeddings_stacks[0].append(TreeEmbedding(current_logic, True))

                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                    current_logic_left_childs.append(current_logic_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                    current_logic_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
                current_logic_beams.append(
                    TreeBeam(logic_b.score+float(logic_tv), current_logic_stack, current_logic_embeddings_stacks,
                                            current_logic_left_childs, current_logic_out))

        beams_zip = zip(current_beams, current_logic_beams)
        beams_zip = sorted(beams_zip, key=lambda x: x[0].score, reverse=True)
        
        beams = [data[0] for data in beams_zip]
        logic_beams = [data[1] for data in beams_zip]

        beams = beams[:beam_size]
        logic_beams = logic_beams[:beam_size]

        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out, logic_beams[0]
