from operator import gt
import random
import json
import copy
import re
import nltk
import torch
import jieba
import jieba.posseg as pseg
import numpy as np
from copy import deepcopy

from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertModel

PAD_token = 0

class MathWP_Dataset(Dataset):
    def __init__(self, data_pairs): 
        super().__init__()
        self.data = data_pairs
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class Lang:
    """
    class to save the vocab and two dict: the word->index and index->word
    """
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count word tokens
        self.num_start = 0 # start index of nums and generate nums

    def add_sen_to_vocab(self, sentence):  # add words of sentence to vocab
        for word in sentence:
            pattern = "N\d+|\[NUM\]|\d+"
            if re.search(pattern, word): # 跳过数字,仅包含 + - * / 
                continue
            if word not in self.index2word:
                self.word2index[word] = self.n_words
                self.word2count[word] = 1
                self.index2word.append(word)
                self.n_words += 1
            else:
                self.word2count[word] += 1

    def trim(self, min_count):  # trim words betlow a certain count threshold
        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.index2word), len(keep_words) / len(self.index2word)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = []
        self.n_words = 0  # Count default tokens

        for word in keep_words:
            self.word2index[word] = self.n_words
            self.index2word.append(word)
            self.n_words += 1

    def build_input_lang(self, trim_min_count):  # build the input lang vocab and dict
        if trim_min_count > 0:
            self.trim(trim_min_count)
            self.index2word = ["[PAD]", "[NUM]", "[UNK]"] + self.index2word
        else:
            self.index2word = ["[PAD]", "[NUM]"] + self.index2word
        self.word2index = {}
        self.n_words = len(self.index2word)
        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

    # def build_output_lang(self, generate_num, copy_nums):  # build the output lang vocab and dict
    #     self.index2word = ["[PAD]", "EOS"] + self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] +\
    #                       ["SOS", "[UNK]"]
    #     self.n_words = len(self.index2word)
    #     for i, j in enumerate(self.index2word):
    #         self.word2index[j] = i

    def build_output_lang_for_tree(self, generate_num, copy_nums):  # build the output lang vocab and dict
        self.num_start = len(self.index2word)

        self.index2word = self.index2word + generate_num + ["N" + str(i) for i in range(copy_nums)] + ["[UNK]"]
        self.n_words = len(self.index2word)

        for i, j in enumerate(self.index2word):
            self.word2index[j] = i

"""
data格式：
{
    "id":"1",
    "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
    "segmented_text":"镇海 雅乐 学校 二年级 的 小朋友 到 一条 小路 的 一边 植树 ． 小朋友 们 每隔 2 米 种 一棵树 （ 马路 两头 都 种 了 树 ） ， 最后 发现 一共 种 了 11 棵 ， 这 条 小路 长 多少 米 ．",
    "equation":"x=(11-1)*2",
    "ans":"20"
}
"""
def load_raw_data(filename, linenum=7):  # load the json data to list(dict()) for MATH 23K
    print("Reading lines...")
    f = open(filename, encoding="utf-8")
    js = ""
    data = []
    for i, s in enumerate(f):
        js += s
        i += 1
        if i % linenum == 0:  # every [linenum] line is a json
            data_d = json.loads(js)
            if "千米/小时" in data_d["equation"]:
                data_d["equation"] = data_d["equation"][:-5]
            data.append(data_d)
            js = ""

    return data

def read_json(filename):
    with open(filename,'r') as f:
        json_data = json.load(f)
    return json_data

def write_json(filename, data):
    with open(filename,'w') as f:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        f.write(json_data)

# 对表达式进行前序遍历
def from_infix_to_prefix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    expression = deepcopy(expression)
    expression.reverse()
    for e in expression:
        if e in [")", "]"]:
            st.append(e)
        elif e == "(":
            c = st.pop()
            while c != ")":
                res.append(c)
                c = st.pop()
        elif e == "[":
            c = st.pop()
            while c != "]":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def cls_tokenize(tokenizer, text):
    # return ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
    return tokenizer.tokenize(text)

def get_op_mask(logic_path, output_lang):
    logic_dict = read_json(logic_path)
    logic_text = [logic_dict[key] for key in logic_dict]
    logic_text = [line.split() for line in logic_text]
    op_mask = []
    for i in range(4):
        op_mask_line = []
        op = output_lang.index2word[i]
        for text in logic_text:
            if len(text) == 1 or text[3] == op:
                op_mask_line.append(1)
            else:
                op_mask_line.append(0)
        op_mask.append(op_mask_line)
    op_mask.append([1]*len(op_mask[0]))
    return op_mask

def tokenize_logic(logic_path, tokenizer, ignore_commonsense=True):
    logic_dict = read_json(logic_path)
    logic_text = [logic_dict[key] for key in logic_dict]
    
    # ignore commonsense
    if ignore_commonsense:
        logic_text = logic_text[1:] 
        
    logic_tokens = [cls_tokenize(tokenizer, line) for line in logic_text] 
    logic_token_ids = [tokenizer.convert_tokens_to_ids(line) for line in logic_tokens]
    logic_attention_mask = []
    logic_token_type_ids = []
    max_length = max([len(line) for line in logic_token_ids])
    for i in range(len(logic_token_ids)):
        padding_idx = [0]*(max_length-len(logic_token_ids[i]))
        logic_attention_mask.append([1]*len(logic_token_ids[i])+padding_idx)
        logic_token_ids[i] += padding_idx
        logic_token_type_ids.append([0]*max_length)
    return logic_token_ids, logic_attention_mask, logic_token_type_ids


def create_interpretation(root, output_prefix, idx):
    inter = {
        "logic": "",
        "op": "",
        "left": {},
        "right": {}
    } 
    # print(idx)
    # print(output_prefix[idx], output_prefix)
    if output_prefix[idx] in ['+', '-', '*', '/', '^']:
        inter["logic"] = ""
        inter["op"] = output_prefix[idx]
        # print("left")
        left_tree = create_interpretation(root, output_prefix, idx+1)
        inter["left"] = left_tree[0]
        # print("right")
        right_tree = create_interpretation(root, output_prefix, left_tree[1])
        inter["right"] = right_tree[0]
        # print(root)
        return (inter, right_tree[1])
    else: 
        inter["logic"] = "0"
        inter["op"] = output_prefix[idx]
        # print("LEAF")
        # print(root)
        return (inter, idx+1)


def indexes_from_sentence_output(lang, sentence, tree=False):
    res = []
    idx = 0
    for word in sentence:
        if len(word) == 0:
            print("##wrong output:", sentence)
            continue
        if word in lang.word2index:
            res.append(lang.word2index[word])
            idx += 1
        else:
            res.append(lang.word2index["[UNK]"])
            print("##output got [UNK]! :", sentence)
            exit()
            idx += 1
        
    if "EOS" in lang.index2word and not tree:
        res.append(lang.word2index["EOS"])
    return res

def find_output_prefix(interpretation, output_prefix):
    if interpretation == {}:
        return
    output_prefix.append(interpretation["op"])
    find_output_prefix(interpretation["left"], output_prefix)
    find_output_prefix(interpretation["right"], output_prefix)

def find_inter_prefix(interpretation, inter_prefix):
    if interpretation == {}:
        return
    inter_prefix.append(int(interpretation["logic"]))
    find_inter_prefix(interpretation["left"], inter_prefix)
    find_inter_prefix(interpretation["right"], inter_prefix)


def word_is_ignore(words, ignore):
    for ig in ignore:
        if ig in words:
            return True
    return False

def transfer_num_retriever(data, tokenizer, mask=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore_ch = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．", 
        "+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", 
        "N", "NUM"]

    pairs = []
    generate_nums = ['1', '2', '3.14', '4']
    generate_nums_dict = {'1':0, '2':0, '3.14':0, '4':0}
    count_num_max = 0 # 一句话里最多有多少个[NUM]

    for line in data:
        # 可解释性数据接口
        # interpretation = {}# create_interpretation({}, out_seq_prefix, 0)[0]
        ## *** 对数据集有一个假设:
        ##      equation中数字的形式和文本一样,不然该函数无法提取equation中的数字
        
        nums = [] # 按顺序记录文本中所有出现的数字（可重复）--str
        nums_fraction = [] # 只记录文本中出现的分数（可重复）

        s = line["original_text"]
        # jieba会把 单空格 自动分出一个token，利用此特性可方便的做出NUM mask
        pos = re.search(pattern, s)
        while(pos):
            nums.append(s[pos.start():pos.end()])
            s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
            pos = re.search(pattern, s)
        # mask_seq = s.split(' ') 
        
        if mask:    
            pseg_attr = pseg.cut(s,use_paddle=True) #paddle模式
            seg = jieba.tokenize(s)
            seg = [t[0] for t in seg]
            
            words_attr = {}
            for word, flag in pseg_attr:   
                words_attr[word] = flag

            mask_list = []
            for w in seg:
                if w in words_attr and words_attr[w] in attr_use and not word_is_ignore(w, ignore_ch):
                    mask_list.append('[MASK]')
                else:
                    mask_list.append(w)
            
            mask_text = ''.join(mask_list)
            s = mask_text

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        
        tokens = [] # 最终 tokens
        match_list = []
        num_token = '[num]' # 被tokenizer处理为小写了
        tokens_initial = tokenizer.tokenize(s) # 先使用tokenizer进行初步切割
        # 假设: 被tokenizer切割的'[NUM]'token, 必定以'['开始,以']'结束
        for t in tokens_initial:
            match_text = ''.join(match_list).replace('#', '')
            text_now = match_text + t.replace('#', '')
            if text_now == num_token: # 完全匹配则直接加入
                tokens.append('[NUM]')
                match_list = []
            elif num_token.startswith(text_now): # 匹配前缀
                match_list.append(t)
            else:
                tokens += match_list
                match_list = []
                if num_token.startswith(t.replace('#', '')):
                    match_list.append(t)
                else:
                    tokens.append(t)

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1: # 如果只出现一次
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        count_num_max = max(count_num_max, len(nums))

        num_idx = []
        for i, j in enumerate(tokens):
            if j == "[NUM]":
                num_idx.append(i)
        assert len(nums) == len(num_idx)
        
        # assert len(line["output_prefix"].split(" ")) <= 13

        pairs.append({
            'original_text':line["original_text"].strip().split(" "), 
            'tokens': tokens, # 对数字改为[NUM]后的token列表
            'output': [], #line["output_prefix"].split(" "),# 先序遍历后的表达式
            'nums': nums, # 按顺序记录出现的数字 
            'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
            'id':line['id'],
            'interpretation': [], #line["interpretation"], # 可解释性标注
            'inter_prefix': [], #inter_prefix, # 可解释性标注的前序遍历logic
            'inter_mask': [], #inter_mask, # 前序遍历顺序logic输出位置的idx
            'inter_multi_label': [], #inter_multi_label, # 一个问题所用到的logic多标签
        })
    return pairs, generate_nums, count_num_max

def process_data_pipeline_retriever(train_data_path, valid_data_path, test_data_path, tokenizer, 
                            debug=False, prompt_text_idx=None, mask=False):
    train_data = read_json(train_data_path)
    valid_data = read_json(valid_data_path)
    if test_data_path:
        test_data = read_json(test_data_path)
    else:
        test_data = []

    if prompt_text_idx != None:
        train_data = merge_prompt_text(train_data, prompt_text_idx["train"])
        valid_data = merge_prompt_text(valid_data, prompt_text_idx["valid"])
        test_data = merge_prompt_text(test_data, prompt_text_idx["test"])
    
    if debug:
        train_data = train_data[:1000]
        valid_data = valid_data[:200]
        test_data = test_data[:200]

    train_data, generate_nums, copy_nums = transfer_num_retriever(
        train_data, 
        tokenizer,  
        mask=mask,
    )
    valid_data, _, _ = transfer_num_retriever(
        valid_data, 
        tokenizer,  
        mask=mask,
    )
    test_data, _, _ = transfer_num_retriever(
        test_data, 
        tokenizer,  
        mask=mask,
    )

    # ignore_list = [ignore_list_train, ignore_list_valid, ignore_list_test]
    return train_data, valid_data, test_data, generate_nums, copy_nums#, ignore_list

"""
面向数据:(仅考虑最基础的数据) # 中英文均可
{
    "id":"1",
    "original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
    "equation":"x=(11-1)*2",
    "ans":"20"
}

输出数据:
{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
}
"""
def transfer_num(data, tokenizer, logic_path=None, mask=False, gt=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore_ch = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．", 
        "+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", 
        "N", "NUM"]

    pairs = []
    generate_nums = ['1', '2', '3.14', '4']
    generate_nums_dict = {'1':0, '2':0, '3.14':0, '4':0}
    count_num_max = 0 # 一句话里最多有多少个[NUM]
    logic_text_size = 210

    if logic_path != None:
        logic_dict = read_json(logic_path)
        logic_text = [logic_dict[key] for key in logic_dict]
        logic_text_size = len(logic_text)

    for line in data:
        # 可解释性数据接口
        # interpretation = {}# create_interpretation({}, out_seq_prefix, 0)[0]
        inter_prefix = list()
        find_inter_prefix(line["interpretation"], inter_prefix)
        inter_mask = [0 if i == -1 else 1 for i in inter_prefix]
        inter_prefix = [0 if i == -1 else i for i in inter_prefix] # original not mask

        inter_multi_label = [0]*logic_text_size
        for mask, prefix in zip(inter_mask, inter_prefix):
            if mask == 1:
                inter_multi_label[prefix] = 1
        inter_multi_label = inter_multi_label[1:]
        ## *** 对数据集有一个假设:
        ##      equation中数字的形式和文本一样,不然该函数无法提取equation中的数字
        
        nums = [] # 按顺序记录文本中所有出现的数字（可重复）--str
        nums_fraction = [] # 只记录文本中出现的分数（可重复）

        s = line["original_text"]
        # jieba会把 单空格 自动分出一个token，利用此特性可方便的做出NUM mask
        pos = re.search(pattern, s)
        while(pos):
            nums.append(s[pos.start():pos.end()])
            s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
            pos = re.search(pattern, s)
        # mask_seq = s.split(' ') 
        
        if mask:    
            pseg_attr = pseg.cut(s,use_paddle=True) #paddle模式
            seg = jieba.tokenize(s)
            seg = [t[0] for t in seg]
            
            words_attr = {}
            for word, flag in pseg_attr:   
                words_attr[word] = flag

            mask_list = []
            for w in seg:
                if w in words_attr and words_attr[w] in attr_use and not word_is_ignore(w, ignore_ch):
                    mask_list.append('[MASK]')
                else:
                    mask_list.append(w)
            
            mask_text = ''.join(mask_list)
            s = mask_text

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
        
        if logic_path != None:
            if "prompt_text_idx" not in line:
                prompt_text_idx = inter_prefix
            else:
                prompt_text_idx = line["prompt_text_idx"]
            
            if gt:
                prompt_text_idx = list()
                for i in range(len(inter_multi_label)):
                    if inter_multi_label[i] == 1:
                       prompt_text_idx.append(i+1) 

            s_logic = str()
            for idx in prompt_text_idx:
                if idx != 0:
                    s_logic += logic_text[idx] + ','
            s_logic = '[' + s_logic + ']'

        tokens = [] # 最终 tokens
        match_list = []
        num_token = '[num]' # 被tokenizer处理为小写了
        tokens_initial = tokenizer.tokenize(s) # 先使用tokenizer进行初步切割
        # 假设: 被tokenizer切割的'[NUM]'token, 必定以'['开始,以']'结束
        for t in tokens_initial:
            match_text = ''.join(match_list).replace('#', '')
            text_now = match_text + t.replace('#', '')
            if text_now == num_token: # 完全匹配则直接加入
                tokens.append('[NUM]')
                match_list = []
            elif num_token.startswith(text_now): # 匹配前缀
                match_list.append(t)
            else:
                tokens += match_list
                match_list = []
                if num_token.startswith(t.replace('#', '')):
                    match_list.append(t)
                else:
                    tokens.append(t)

        if logic_path != None:
            logic_tokens = tokenizer.tokenize(s_logic) # 先使用tokenizer进行初步切割
            tokens = ['[CLS]'] + tokens + logic_tokens + ['[SEP]']
            # tokens = ['[CLS]'] + logic_tokens + tokens + ['[SEP]']
        else:
            tokens = ['[CLS]'] + tokens + ['[SEP]']

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1: # 如果只出现一次
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        count_num_max = max(count_num_max, len(nums))

        for s in line["output_prefix"].split(" "):  # tag the num which is generated(文本中未出现的常量)
            if s[0].isdigit() and s not in generate_nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_idx = []
        for i, j in enumerate(tokens):
            if j == "[NUM]":
                num_idx.append(i)
        assert len(nums) == len(num_idx)
        if nums != line["nums"].split(' '):
            print("nums error: ", line["id"])
        
        assert len(line["output_prefix"].split(" ")) <= 13

        pairs.append({
            'original_text':line["original_text"].strip().split(" "), 
            'tokens': tokens, # 对数字改为[NUM]后的token列表
            'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
            'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
            'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
            'id':line['id'],
            'interpretation': line["interpretation"], # 可解释性标注
            'inter_prefix':inter_prefix, # 可解释性标注的前序遍历logic
            'inter_mask': inter_mask, # 前序遍历顺序logic输出位置的idx
            'inter_multi_label': inter_multi_label, # 一个问题所用到的logic多标签
            'prompt_text_idx': [] if logic_path == None else prompt_text_idx ,
        })

    return pairs, generate_nums, count_num_max

def merge_prompt_text(data, prompt_text_idx):
    for line, p_idx in zip(data, prompt_text_idx):
        line["prompt_text_idx"] = p_idx
    return data
"""
面向数据:(仅考虑最基础的数据) # 中英文均适配
{
"id":"1",
"original_text":"镇海雅乐学校二年级的小朋友到一条小路的一边植树．小朋友们每隔2米种一棵树（马路两头都种了树），最后发现一共种了11棵，这条小路长多少米．",
"equation":"x=(11-1)*2",
"ans":"20"
}

输出数据:
{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
}
"""
def process_data_pipeline(train_data_path, valid_data_path, test_data_path, tokenizer, 
                debug=False, prompt_text_idx=None, logic_path=None, mask=False, gt=False):
    train_data = read_json(train_data_path)
    valid_data = read_json(valid_data_path)
    if test_data_path:
        test_data = read_json(test_data_path)
    else:
        test_data = []

    if prompt_text_idx != None:
        train_data = merge_prompt_text(train_data, prompt_text_idx["train"])
        valid_data = merge_prompt_text(valid_data, prompt_text_idx["valid"])
        test_data = merge_prompt_text(test_data, prompt_text_idx["test"])
    
    if debug:
        train_data = train_data[:1000]
        valid_data = valid_data[:200]
        test_data = test_data[:200]

    train_data, generate_nums, copy_nums = transfer_num(
        train_data, 
        tokenizer,  
        logic_path=logic_path, 
        mask=mask,
        gt=gt,
    )
    valid_data, _, _ = transfer_num(
        valid_data, 
        tokenizer,  
        logic_path=logic_path, 
        mask=mask,
        gt=gt,
    )
    test_data, _, _ = transfer_num(
        test_data, 
        tokenizer,  
        logic_path=logic_path, 
        mask=mask,
        gt=gt,
    )

    # ignore_list = [ignore_list_train, ignore_list_valid, ignore_list_test]
    return train_data, valid_data, test_data, generate_nums, copy_nums#, ignore_list
    

def transfer_num2(data, tokenizer, logic_path, mask=False):
    pattern = re.compile("\d*\(\d+/\d+\)\d*|\d+\.\d+%?|\d+%?")
    attr_use = ["n", "t", "nr", "ns", "nt", "PER", "LOC", "TIME"]
    ignore_ch = [",", "，", ".", "。", "?", "？", "!", "！", ":", "：", "、", ";", "；", "．", 
        "+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", 
        "N", "NUM"]

    pairs = []
    generate_nums = ['1', '2', '3.14', '4']
    generate_nums_dict = {'1':0, '2':0, '3.14':0, '4':0}
    count_num_max = 0 # 一句话里最多有多少个[NUM]

    logic_dict = read_json(logic_path)
    logic_text = [logic_dict[key] for key in logic_dict]
    logic_text_size = len(logic_text)

    for line in data:
        # 可解释性数据接口
        # interpretation = {}# create_interpretation({}, out_seq_prefix, 0)[0]
        inter_prefix = list()
        find_inter_prefix(line["interpretation"], inter_prefix)
        inter_mask = [0 if i == -1 else 1 for i in inter_prefix]
        inter_prefix = [i+1 for i in inter_prefix] # original not mask

        inter_multi_label = [0]*logic_text_size
        for prefix in inter_prefix:
            inter_multi_label[prefix] = 1
        inter_multi_label[0:2] = [0, 0]
        ## *** 对数据集有一个假设:
        ##      equation中数字的形式和文本一样,不然该函数无法提取equation中的数字
        
        nums = [] # 按顺序记录文本中所有出现的数字（可重复）--str
        nums_fraction = [] # 只记录文本中出现的分数（可重复）

        s = line["original_text"]
        # jieba会把 单空格 自动分出一个token，利用此特性可方便的做出NUM mask
        pos = re.search(pattern, s)
        while(pos):
            nums.append(s[pos.start():pos.end()])
            s = s[:pos.start()] + ' [NUM] ' + s[pos.end():] # 将数字改写为[NUM] token
            pos = re.search(pattern, s)
        # mask_seq = s.split(' ') 
        
        if mask:    
            pseg_attr = pseg.cut(s,use_paddle=True) #paddle模式
            seg = jieba.tokenize(s)
            seg = [t[0] for t in seg]
            
            words_attr = {}
            for word, flag in pseg_attr:   
                words_attr[word] = flag

            mask_list = []
            for w in seg:
                if w in words_attr and words_attr[w] in attr_use and not word_is_ignore(w, ignore_ch):
                    mask_list.append('[MASK]')
                else:
                    mask_list.append(w)
            
            mask_text = ''.join(mask_list)
            s = mask_text

        for num in nums:
            if re.search("\d*\(\d+/\d+\)\d*", num):
                nums_fraction.append(num)
        nums_fraction = sorted(nums_fraction, key=lambda x: len(x), reverse=True)
    
        if "prompt_text_idx" not in line:
            prompt_text_idx = inter_prefix
        else:
            prompt_text_idx = line["prompt_text_idx"]

        s_logic = str()
        for idx in prompt_text_idx:
            if idx != 0:
                s_logic += logic_text[idx+1] + ','
        s_logic = '[' + s_logic + ']'

        tokens = [] # 最终 tokens
        match_list = []
        num_token = '[num]' # 被tokenizer处理为小写了
        tokens_initial = tokenizer.tokenize(s) # 先使用tokenizer进行初步切割
        # 假设: 被tokenizer切割的'[NUM]'token, 必定以'['开始,以']'结束
        for t in tokens_initial:
            match_text = ''.join(match_list).replace('#', '')
            text_now = match_text + t.replace('#', '')
            if text_now == num_token: # 完全匹配则直接加入
                tokens.append('[NUM]')
                match_list = []
            elif num_token.startswith(text_now): # 匹配前缀
                match_list.append(t)
            else:
                tokens += match_list
                match_list = []
                if num_token.startswith(t.replace('#', '')):
                    match_list.append(t)
                else:
                    tokens.append(t)

        logic_tokens = tokenizer.tokenize(s_logic) # 先使用tokenizer进行初步切割
        tokens = ['[CLS]'] + tokens + logic_tokens + ['[SEP]']
        # tokens = ['[CLS]'] + logic_tokens + tokens + ['[SEP]']

        def seg_and_tag(st):  # seg the equation and tag the num
            res = []
            for n in nums_fraction:
                if n in st:
                    p_start = st.find(n)
                    p_end = p_start + len(n)
                    if p_start > 0:
                        res += seg_and_tag(st[:p_start])
                    if nums.count(n) == 1: # 如果只出现一次
                        res.append("N"+str(nums.index(n)))
                    else:
                        res.append(n)
                    if p_end < len(st):
                        res += seg_and_tag(st[p_end:])
                    return res
            pos_st = re.search("\d+\.\d+%?|\d+%?", st)
            if pos_st:
                p_start = pos_st.start()
                p_end = pos_st.end()
                if p_start > 0:
                    res += seg_and_tag(st[:p_start])
                st_num = st[p_start:p_end]
                if nums.count(st_num) == 1:
                    res.append("N"+str(nums.index(st_num)))
                else:
                    res.append(st_num)
                if p_end < len(st):
                    res += seg_and_tag(st[p_end:])
                return res
            for ss in st:
                res.append(ss)
            return res

        count_num_max = max(count_num_max, len(nums))

        for s in line["output_prefix"].split(" "):  # tag the num which is generated(文本中未出现的常量)
            if s[0].isdigit() and s not in generate_nums:
                generate_nums.append(s)
                generate_nums_dict[s] = 0
            if s in generate_nums and s not in nums:
                generate_nums_dict[s] = generate_nums_dict[s] + 1

        num_idx = []
        for i, j in enumerate(tokens):
            if j == "[NUM]":
                num_idx.append(i)
        assert len(nums) == len(num_idx)
        if nums != line["nums"].split(' '):
            print("nums error: ", line["id"])
        
        assert len(line["output_prefix"].split(" ")) <= 13

        pairs.append({
            'original_text':line["original_text"].strip().split(" "), 
            'tokens': tokens, # 对数字改为[NUM]后的token列表
            'output': line["output_prefix"].split(" "),# 先序遍历后的表达式
            'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
            'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
            'id':line['id'],
            'interpretation': line["interpretation"], # 可解释性标注
            'inter_prefix':inter_prefix, # 可解释性标注的前序遍历logic
            'inter_mask': inter_mask, # 前序遍历顺序logic输出位置的idx
            'inter_multi_label': inter_multi_label, # 一个问题所用到的logic多标签
        })

    return pairs, generate_nums, count_num_max


def process_data_pipeline2(train_data_path, valid_data_path, test_data_path, tokenizer, 
                            debug=False, prompt_text_idx=None, logic_path=None, mask=False):
    train_data = read_json(train_data_path)
    valid_data = read_json(valid_data_path)
    if test_data_path:
        test_data = read_json(test_data_path)
    else:
        test_data = []

    if prompt_text_idx != None:
        train_data = merge_prompt_text(train_data, prompt_text_idx["train"])
        valid_data = merge_prompt_text(valid_data, prompt_text_idx["valid"])
        test_data = merge_prompt_text(test_data, prompt_text_idx["test"])
    
    if debug:
        train_data = train_data[:1000]
        valid_data = valid_data[:200]
        test_data = test_data[:200]

    train_data, generate_nums, copy_nums = transfer_num2(
        train_data, 
        tokenizer,  
        logic_path=logic_path, 
        mask=mask,
    )
    valid_data, _, _ = transfer_num2(
        valid_data, 
        tokenizer,  
        logic_path=logic_path, 
        mask=mask,
    )
    test_data, _, _ = transfer_num2(
        test_data, 
        tokenizer,  
        logic_path=logic_path, 
        mask=mask,
    )

    # ignore_list = [ignore_list_train, ignore_list_valid, ignore_list_test]
    return train_data, valid_data, test_data, generate_nums, copy_nums#, ignore_list


"""
data格式:{
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'interpretation': line["interpretation"], # 可解释性标注
    'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
    'id':line['id'],
    pair["token_ids"] = token_ids
    pair["token_type_ids"] = token_type_ids
    pair["attention_mask"] = attention_mask
    pair["output"] = output_cell
}
"""
def prepare_pairs(pairs, output_lang, tokenizer, max_seq_length, tree=False):
    PAD_id = tokenizer.pad_token_id
    processed_pairs = []
    # ignore_input_len = 0
    """
    pair:{
        'original_text':line["original_text"].strip().split(" "),
        'tokens': tokens, # 对数字改为[NUM]后的token列表
        'output': line["output_prefix"].split(" "),# 中序遍历后的表达式
        'nums': line["nums"].split(' '), # 按顺序记录出现的数字 
        'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
        'id':line['id'],
        'interpretation': line["interpretation"], # 可解释性标注
        'inter_prefix':inter_prefix,# 可解释性标注的前序遍历logic
    }
    """
    for pair in pairs:
        ## 先处理num_stack
        num_stack = []
        for word in pair['output']:  # 处理表达式
            temp_num = []
            flag_not = True
            if word not in output_lang.index2word:
                print("output lang not find: ", word, '||', pair['output'], pair["id"])     
                flag_not = False
                for i, j in enumerate(pair['nums']):
                    if j == word:
                        temp_num.append(i)

            if not flag_not and len(temp_num) != 0:
                num_stack.append(temp_num)
            if not flag_not and len(temp_num) == 0:
                num_stack.append([_ for _ in range(len(pair["nums"]))])

        num_stack.reverse()
        assert num_stack == []
        # 忽略长度＞180的问题, 此处用ignore控制的原因在于：文本过长容易导致内存错误
        assert len(pair["tokens"]) <= max_seq_length

        output_cell = indexes_from_sentence_output(output_lang, pair['output'], tree=tree)   

        token_ids = tokenizer.convert_tokens_to_ids(pair["tokens"])
        token_len = len(token_ids)
        # Padding 
        padding_ids = [PAD_id]*(max_seq_length - len(token_ids))
        token_ids += padding_ids
        # token_type_ids
        token_type_ids = [0]*max_seq_length
        # attention_mask
        attention_mask = [1]*token_len + padding_ids
        
        ### Testing num 
        for idx in pair["num_idx"]:
            assert pair["tokens"][idx] == '[NUM]'
        
        pair["token_ids"] = token_ids
        pair["token_type_ids"] = token_type_ids
        pair["attention_mask"] = attention_mask
        pair["output"] = output_cell
        pair["nums"] = pair["nums"]
        pair["id"] = pair["id"]
        pair["original_text"] = pair["original_text"]
        
        processed_pairs.append(pair)

    return processed_pairs#, ignore_input_len


def prepare_bert_data(pairs_train, pairs_valid, pairs_test, generate_nums, 
                      copy_nums, tokenizer, max_seq_length,tree=True):
    output_lang = Lang()
    """
    {
    'original_text':line["original_text"].strip().split(" "),
    'tokens': tokens, # 对数字改为[NUM]后的token列表
    'output': out_seq_prefix, # 先序遍历后的表达式
    'nums': nums, # 按顺序记录出现的数字 
    'num_idx': num_idx, # 记录input列表中出现[NUM]的下标
    'id':line['id'],
    # 'interpretation': interpretation, # 可解释性标注
    }
    """
    ## build lang
    print("Tokenizing/Indexing words...")
    for pair in pairs_train:
        output_lang.add_sen_to_vocab(pair['output'])

    if tree:
        output_lang.build_output_lang_for_tree(generate_nums, copy_nums)
    else:
        output_lang.build_output_lang(generate_nums, copy_nums)
    print(output_lang.index2word)
    print('Indexed %d words in output' % (output_lang.n_words))

    train_pairs = prepare_pairs(pairs_train, output_lang, tokenizer, max_seq_length, tree)
    valid_pairs = prepare_pairs(pairs_valid, output_lang, tokenizer, max_seq_length, tree)
    test_pairs = prepare_pairs(pairs_test, output_lang, tokenizer, max_seq_length, tree)

    print('Number of training data %d' % (len(train_pairs)))
    print('Number of validating data %d' % (len(valid_pairs)))
    print('Number of testing data %d' % (len(test_pairs)))
    
    # ignore_len_list = [ignore_input_len_train, ignore_input_len_valid, ignore_input_len_test]
    return output_lang, train_pairs, valid_pairs, test_pairs#, ignore_len_list


"""
data_batch = {
    "max_token_len": max(token_len),
    "token_ids": [line[:max(token_len)] for line in token_ids],
    "token_type_ids": [line[:max(token_len)] for line in token_type_ids],
    "attention_mask": [line[:max(token_len)] for line in attention_mask],
    "output": pad_output_seq(output, max(output_len), 0),
    "output_len": output_len,
    "inter": pad_output_seq(inter, max(output_len), -1),
    "nums": nums,
    "num_size": num_size,
    "num_idx": num_idx,
}
"""

def pad_output_seq(seq, max_length, PAD_token):
    # PAD_token = 0
    seq = [line+[PAD_token for _ in range(max_length-len(line))] for line in seq]
    return seq

def my_collate(batch_line):
    batch_line = deepcopy(batch_line)
    token_len = []
    token_ids = []
    token_type_ids = []
    attention_mask = []
    output = []
    output_len = []
    inter_prefix = []
    inter_mask = []
    inter_multi_label = []
    nums = []
    num_size = []
    num_idx = []
    ids = []

    for line in batch_line:
        token_len.append(len(line["tokens"]))
        token_ids.append(line["token_ids"])
        token_type_ids.append(line["token_type_ids"])
        attention_mask.append(line["attention_mask"])
        output.append(line["output"])
        output_len.append(len(line["output"]))
        inter_prefix.append(line["inter_prefix"])
        inter_mask.append(line["inter_mask"])
        inter_multi_label.append(line['inter_multi_label'])
        nums.append(line["nums"])
        num_size.append(len(line["nums"]))
        num_idx.append(line["num_idx"])
        ids.append(line["id"])

    batch = {
        "max_token_len": max(token_len),
        "token_ids": [line[:max(token_len)] for line in token_ids],
        "token_type_ids": [line[:max(token_len)] for line in token_type_ids],
        "attention_mask": [line[:max(token_len)] for line in attention_mask],
        "output": pad_output_seq(output, max(output_len), 0),
        "output_len":output_len,
        "inter_prefix": pad_output_seq(inter_prefix, max(output_len), 0), # original padding_idx=-1
        'inter_mask': pad_output_seq(inter_mask, max(output_len), 0),
        'inter_multi_label': inter_multi_label,
        "nums": nums,
        "num_size":num_size,
        "num_idx": num_idx,
        "id": ids,
    }
    return batch


def my_collate_logic_decoder(batch_line):
    batch_line = deepcopy(batch_line)
    token_len = []
    token_ids = []
    token_type_ids = []
    attention_mask = []
    output = []
    output_len = []
    inter_prefix = []
    inter_mask = []
    goal = []
    ids = []
    logic_op_mask = []

    for line in batch_line:
        token_len.append(len(line["tokens"]))
        token_ids.append(line["token_ids"])
        token_type_ids.append(line["token_type_ids"])
        attention_mask.append(line["attention_mask"])
        output.append(line["output"])
        output_len.append(len(line["output"]))
        inter_prefix.append(line["inter_prefix"])
        inter_mask.append(line["inter_mask"])
        goal.append(line["goal_stack"])
        ids.append(line["id"])
        logic_op_mask.append(line["logic_op_mask"])

    batch = {
        "max_token_len": max(token_len),
        "token_ids": [line[:max(token_len)] for line in token_ids],
        "token_type_ids": [line[:max(token_len)] for line in token_type_ids],
        "attention_mask": [line[:max(token_len)] for line in attention_mask],
        "output": pad_output_seq(output, max(output_len), 0),
        "output_len":output_len,
        "inter_prefix": pad_output_seq(inter_prefix, max(output_len), 0), # original padding_idx=-1
        'inter_mask': pad_output_seq(inter_mask, max(output_len), 0),
        'goal': pad_goal(goal, max(output_len)),
        "id": ids,
        "logic_op_mask": pad_op_mask(logic_op_mask, max(output_len)),
    }
    return batch

def pad_op_mask(logic_op_mask, length):
    for i in range(len(logic_op_mask)):
        while len(logic_op_mask[i]) < length:
            logic_op_mask[i].append([1]*len(logic_op_mask[i][0]))
    return logic_op_mask

def pad_goal(goal, length):
    for i in range(len(goal)):
        goal[i] = torch.nn.functional.pad(goal[i], pad=(0, 0, 0, length-goal[i].shape[0]), mode='constant', value=0)
    goal = torch.stack(goal, dim=0)
    return goal

def logic_op_2_num(logic_path):
    logic = read_json(logic_path)
    logic = [logic[key] for key in logic]

    # + - * /
    # 0 1 2 3
    add = list()
    minus = list()
    dot = list()
    div = list()

    # A = B op C
    for line in logic:
        if line == "常识性步骤":
            add.append(1)
            minus.append(1)
            dot.append(1)
            div.append(1)
        elif line.split()[3] == '+':
            add.append(1)
            minus.append(0)
            dot.append(0)
            div.append(0)
        elif line.split()[3] == '-':
            add.append(0)
            minus.append(1)
            dot.append(0)
            div.append(0)
        elif line.split()[3] == '*':
            add.append(0)
            minus.append(0)
            dot.append(1)
            div.append(0)
        elif line.split()[3] == '/':
            add.append(0)
            minus.append(0)
            dot.append(0)
            div.append(1)

    # [/, -, +, *, num]
    num = [1]*len(div)
    op2num = list([div, minus, add, dot, num])
    op2num_ = copy.deepcopy(op2num) # negative
    for i in range(len(op2num_)):
        for j in range(len(op2num_[0])):
            op2num_[i][j] = 1 - op2num_[i][j]
    return op2num, op2num_