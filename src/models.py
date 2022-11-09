from ast import Str
from turtle import position
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
from packaging import version
from pathlib import Path
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from transformers import BertTokenizer, BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_bert import (
    # BertEmbeddings,
    BertPooler,
    BertEncoder, 
    BaseModelOutputWithPooling, 
    BERT_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC,
    _TOKENIZER_FOR_DOC,
)
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len # 1 x L x 1
        hidden = hidden.repeat(*repeat_dims)  # B x L x 2H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)      
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size) # B*L x 3H
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x L) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x L
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, goal, logic):
        ## goal: B x 1 X H
        ## logic: L x H
        this_batch_size = goal.size(0)

        logic_len = logic.size(0)
        repeat_dims = [1] * goal.dim()
        repeat_dims[1] = logic_len
        goal = goal.repeat(*repeat_dims)  # B x L x H
        
        logic.unsqueeze(0)
        repeat_dims = [1] * goal.dim()
        repeat_dims[0] = this_batch_size
        logic = logic.repeat(*repeat_dims) # B x L x H

        energy_in = torch.cat((goal, logic), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (B x L) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(this_batch_size, logic_len)# B x L
        # attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x L

        return attn_energies # B x L

class Attention_FineGained(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention_FineGained, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, goal, logic):
        ## goal: B x 1 X H
        ## logic: B x L x H
        # print(goal.shape, logic.shape)
        goal = goal.repeat(1, logic.shape[1], 1)  # B x L x H
        # print(goal.shape, logic.shape)
        energy_in = torch.cat((goal, logic), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (B x L) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(logic.shape[0], logic.shape[1])# B x L
        # attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x L

        return attn_energies # B x L

class LogicAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LogicAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)
        self.goal_logic_attn = Attention(hidden_size, hidden_size)

    def forward(self, goal, encoder_outputs, logic_embedding, seq_mask=None):
        goal = goal.transpose(0, 1)
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * goal.dim()
        repeat_dims[0] = max_len
        goal = goal.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((goal, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        attn_energies = attn_energies.unsqueeze(1) # B x 1 x S
        goal_context = attn_energies.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H

        goal_logic_score = self.goal_logic_attn(goal_context, logic_embedding) # B x L
        logic_attn_score = nn.functional.softmax(goal_logic_score, dim=1) # B x L
        goal_with_logic = torch.mm(logic_attn_score, logic_embedding) # B x H
        return goal_with_logic, goal_logic_score

class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S    
        
        return attn_energies.unsqueeze(1) # B x 1 x S

class Encoder_Bert(nn.Module):
    def __init__(self, dropout=0.5, bert_path="", alpha=0.01, num_relations=3):
        super(Encoder_Bert, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        # Load Bert-wwm model
        self.bert_wwm = BertModel.from_pretrained(bert_path)
        print("Load bert model from:", bert_path)
        self.hidden_size = self.config.hidden_size
        self.dropout = dropout
        # self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        # self.gcn = Graph_Module(self.hidden_size, self.hidden_size, self.hidden_size)

    #   input_ids: [[1011223...,102,0,0,0,0]]       B x S
    #   token_type_ids: [[0,0,0,0,0,0,0,0,0,0]]     B x S    [[1]*B]*S  
    #   attention_mask: [[1,1,1,1,1,1,1,1,1,1]]     B x S    [1]*len + [0]*padlen
    #   sentence's word-size: [unpadding size]      B
    def forward(self, input_ids, token_type_ids, attention_mask):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        encoder_outputs, problem_output = self.bert_wwm(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # problem_output = torch.mean(encoder_outputs, dim=1)
        # pade_outputs = output.last_hidden_state
        # problem_output = output.pooler_output
        # problem_output = self.em_dropout(problem_output) # B x H
        # encoder_outputs = encoder_outputs.transpose(0, 1).contiguous() # S x B x H
        return encoder_outputs, problem_output
    
    def savebert(self, save_path):
        torch.save(self.bert_wwm.state_dict(), save_path)

# op_nums in Prediction: ['+', '-', '*', '/']
class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size # size of generate nums
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        # generate nums vector
        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.op_score = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.num_score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        ## Getting Goal Vector
        ## ------------------------------------------------------------------------------------------------------ ##
        current_embeddings = []
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        # Goal Vector
        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node) # B x 1 x H 
        ## ------------------------------------------------------------------------------------------------------ ##

        ## Getting Context Vector 
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask) # B x 1 x S
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H
        ## ------------------------------------------------------------------------------------------------------ ##

        ## Getting embedding_weight
        # embedding_weight is the concatenation of [word vector of generate_nums and the nums embedding in LM's output]
        batch_size = current_embeddings.size(0)
        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x H
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x len(generate_num + num) x H
        # embedding_weight's dropout
        embedding_weight_ = self.dropout(embedding_weight)
        ## ------------------------------------------------------------------------------------------------------ ##

        ## [Goal vector, Context Vector]
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input) # B x 2H
        ## ------------------------------------------------------------------------------------------------------ ##

        ## Getting num_score and op_score
        try:
            # score of generate nums and nums
            num_score = self.num_score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)
        except Exception as e:
            print(embedding_weight_.shape)
            print(mask_nums.shape)
            print(leaf_input.unsqueeze(1).shape)
            print(e)

        # score of operators
        op_score = self.op_score(leaf_input)
        ## ------------------------------------------------------------------------------------------------------ ##

        ##     num_score, op_score, Goal Vector,  Context Vector,  ALL Num embedding
        return num_score, op_score, current_node, current_context, embedding_weight

# op_nums in Prediction: ['+', '-', '*', '/']
class LogicPrediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(LogicPrediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size # size of generate nums
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.logic_score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, padding_hidden, seq_mask, logic_embeddings):
        ## Getting Goal Vector
        ## ------------------------------------------------------------------------------------------------------ ##
        logic_embeddings = logic_embeddings.unsqueeze(0).repeat(encoder_outputs.shape[1],1,1)
        
        current_embeddings = []
        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)
        # Goal Vector
        current_node = torch.stack(current_node_temp)
        current_embeddings = self.dropout(current_node) # B x 1 x H 
        ## ------------------------------------------------------------------------------------------------------ ##

        ## Getting Context Vector 
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask) # B x 1 x S
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H
        ## ------------------------------------------------------------------------------------------------------ ##
        
        ## [Goal vector, Context Vector]
        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input) # B x 2H
        ## ------------------------------------------------------------------------------------------------------ ##
        logic_score = self.logic_score(leaf_input.unsqueeze(1), logic_embeddings)

        return logic_score, current_node, current_context

# op_nums in Prediction: ['+', '-', '*', '/']
class LogicGenerator(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, dropout=0.5):
        super(LogicGenerator, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size

        # Define layers
        self.dropout = nn.Dropout(dropout)

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.logic_score = Score(hidden_size * 2, hidden_size)

    def forward(self, goal, encoder_outputs, seq_mask, logic_embeddings):
        # goal: [B x L x H]
        # encoder_outputs: [B x S x H]
        # logic_embeddings: [E x H]
        ## Getting Goal Vector
        ## ------------------------------------------------------------------------------------------------------ ##
        logic_embeddings = logic_embeddings.unsqueeze(0).repeat(encoder_outputs.shape[1],1,1)
        all_logic_score = []
        # Goal Vector
        for i in range(goal.shape[1]):
            current_node = goal[:,i,:].unsqueeze(1)
            current_embeddings = self.dropout(current_node) # B x 1 x H 
            ## ------------------------------------------------------------------------------------------------------ ##

            ## Getting Context Vector 
            current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask) # B x 1 x S
            current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x H
            ## ------------------------------------------------------------------------------------------------------ ##
            
            ## [Goal vector, Context Vector]
            leaf_input = torch.cat((current_node, current_context), 2)
            leaf_input = leaf_input.squeeze(1)
            leaf_input = self.dropout(leaf_input) # B x 2H
            ## ------------------------------------------------------------------------------------------------------ ##
            logic_score_node = self.logic_score(leaf_input.unsqueeze(1), logic_embeddings)
            all_logic_score.append(logic_score_node.unsqueeze(1))

        all_logic_score = torch.cat(all_logic_score, dim=1)
        return all_logic_score



# op_nums in GenerateNode: ['+', '-', '*', '/', '[NUM]']
class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        # operator vector
        self.embeddings = nn.Embedding(op_nums, embedding_size) 
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_op_label, current_context):
        node_op_embedding = self.embeddings(node_op_label)
        node_op_label = self.em_dropout(node_op_embedding)
        node_embedding = node_embedding.squeeze(1) # B x H
        current_context = current_context.squeeze(1) # B x H
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        concate_all = torch.cat((node_embedding, current_context, node_op_label), 1)
        l_child = torch.tanh(self.generate_l(concate_all))
        l_child_g = torch.sigmoid(self.generate_lg(concate_all))
        r_child = torch.tanh(self.generate_r(concate_all))
        r_child_g = torch.sigmoid(self.generate_rg(concate_all))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_op_embedding

class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, prompt_len):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # initialize prompt embedding from vocab
        self.prompt_len = prompt_len
        prompt_embeddings_weight = self.word_embeddings.weight[2:2+prompt_len].clone().detach()
        self.prompt_embeddings = nn.parameter.Parameter(prompt_embeddings_weight)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # start after [CLS] token
        inputs_embeds[:, 1:1+self.prompt_len, :] = self.prompt_embeddings
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertPromptModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, prompt_len=0, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.prompt_len = prompt_len
        self.embeddings = BertEmbeddings(config, prompt_len)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint="bert-base-uncased",
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class LogicAttn(nn.Module):
    def __init__(self, hidden_size):
        super(LogicAttn, self).__init__()
        self.hidden_size = hidden_size

        self.attn = Attention(hidden_size, hidden_size)

    def forward(self, input_embedding, logic_embedding):
        input_embedding = input_embedding.unsqueeze(1)
        logic_score = self.attn(input_embedding, logic_embedding) # B x L
        # logic_score = nn.Sigmoid(logic_score)
        return logic_score
