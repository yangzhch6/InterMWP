# coding: utf-8
import os
import time
import math
import json
import pprint
import argparse
import pickle
from traceback import print_list
import torch.optim
from tqdm import tqdm
from src.models import *
from src.pre_data import *
from src.train_and_evaluate import *
from src.expressions_transfer import *
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter


def get_new_fold(data,pairs,group):
    new_fold = []
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        new_fold.append(pair)
    return new_fold

def change_num(num):
    new_num = []
    for item in num:
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            new_num.append(value)
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        else:
            new_num.append(float(item))
    return new_num

def find_output_prefix(interpretation, output_prefix):
    if interpretation == {}:
        return
    output_prefix.append(interpretation["op"])
    find_output_prefix(interpretation["left"], output_prefix)
    find_output_prefix(interpretation["right"], output_prefix)

def set_args():
    parser = argparse.ArgumentParser(description = "bert2tree")

    # 训练模型相关参数
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=300)
    parser.add_argument('--embedding_size', type=int, default=128)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_bert', type=float, default=5e-5)
    parser.add_argument('--weight_decay_bert', type=float, default=1e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=15)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 训练控制相关
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--maskN', action='store_true', default=False)

    # 数据相关参数
    parser.add_argument('--train_data_path', type=str, default="data/train.json")
    parser.add_argument('--valid_data_path', type=str, default="data/valid.json")
    parser.add_argument('--test_data_path' , type=str, default="data/test.json")
    parser.add_argument('--test_full_path' , type=str, default="data/test_full.json")

    # 预训练模型路径
    parser.add_argument('--logic_path', type=str, default="data/logic.json")
    parser.add_argument('--bert_path', type=str, default="/data1/yangzhicheng/Data/models/chinese-bert-wwm")
    
    # 存储相关参数
    parser.add_argument('--save_path', type=str, default="model/retriever/SoftMarginLoss")

    args = parser.parse_args()
    return args

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if USE_CUDA:
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def evaluate_result(args, encoder, retriever, data_loader, logfile):
    start = time.time()
    loss_total = 0
    predict = list()
    target = list()
    for valid_batch in tqdm(data_loader):
        loss, predict_score = evaluate_retriever(
            valid_batch['inter_multi_label'], 
            encoder, retriever,
            valid_batch["token_ids"], 
            valid_batch["token_type_ids"], 
            valid_batch["attention_mask"],
            logic_token_ids, 
            logic_attention_mask, 
            logic_token_type_ids
        )
        loss_total += loss
        predict += predict_score
        target += valid_batch["inter_prefix"]
    R_list, P_list, Predict_list = recall_topk_all(predict, target)
    F1_list = dict()
    for key in R_list:
        F1_list[key] = 2*R_list[key]*P_list[key] / (R_list[key]+P_list[key]+1e-9)

    # print("loss:", loss_total / len(predict))
    print(
        "R1: %5f" %(R_list['1']) + \
        " |R2: %5f" %(R_list['2']) + \
        " |R3: %5f" %(R_list['3']) + \
        " |R4: %5f" %(R_list['4']) + \
        " |R5: %5f" %(R_list['5']) + \
        " |R6: %5f" %(R_list['6']) 
    )
    print(
        "P1: %5f" %(P_list['1']) + \
        " |P2: %5f" %(P_list['2']) + \
        " |P3: %5f" %(P_list['3']) + \
        " |P4: %5f" %(P_list['4']) + \
        " |P5: %5f" %(P_list['5']) + \
        " |P6: %5f" %(P_list['6'])
    )
    print(
        "F1_1: %5f" %(F1_list['1']) + \
        " |F1_2: %5f" %(F1_list['2']) + \
        " |F1_3: %5f" %(F1_list['3']) + \
        " |F1_4: %5f" %(F1_list['4']) + \
        " |F1_5: %5f" %(F1_list['5']) + \
        " |F1_6: %5f" %(F1_list['6'])
    )

    print("evaluating time", time_since(time.time() - start))
    print("--------------------------------")

    with open(logfile, 'a') as file_object:
        # file_object.write("loss: %f\n" %(loss_total / len(predict)))
        file_object.write(
            "R1: %5f" %(R_list['1']) + \
            " |R2: %5f" %(R_list['2']) + \
            " |R3: %5f" %(R_list['3']) + \
            " |R4: %5f" %(R_list['4']) + \
            " |R5: %5f" %(R_list['5']) + \
            " |R6: %5f" %(R_list['6'])
        )
        file_object.write(
            "P1: %5f" %(P_list['1']) + \
            " |P2: %5f" %(P_list['2']) + \
            " |P3: %5f" %(P_list['3']) + \
            " |P4: %5f" %(P_list['4']) + \
            " |P5: %5f" %(P_list['5']) + \
            " |P6: %5f" %(P_list['6']) 
        )
        file_object.write(
            "F1_1: %5f" %(F1_list['1']) + \
            " |F1_2: %5f" %(F1_list['2']) + \
            " |F1_3: %5f" %(F1_list['3']) + \
            " |F1_4: %5f" %(F1_list['4']) + \
            " |F1_5: %5f" %(F1_list['5']) + \
            " |F1_6: %5f" %(F1_list['6']) 
        )
        file_object.write("evaluating time " + str(time_since(time.time() - start)) + "\n")
        file_object.write("--------------------------------\n")
    
    # 存储时，需要对predict里的每一个id加1
    for key in Predict_list:
        Predict_list[key] = [[idx+1 for idx in line] for line in Predict_list[key]]

    return R_list, P_list, F1_list, Predict_list

if __name__ == "__main__":
    args = set_args()
    #创建save文件夹
    print("** save path:", args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("make dir ", args.save_path)
    
    log_writer = SummaryWriter()

    setup_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    logic_token_ids, logic_attention_mask, logic_token_type_ids = tokenize_logic(
        args.logic_path, 
        tokenizer,
    )
    train_fold, valid_fold, test_fold, generate_nums, copy_nums = process_data_pipeline(
        args.train_data_path, args.valid_data_path, args.test_data_path, tokenizer,
        debug=args.debug,
        logic_path=args.logic_path,
        mask=args.maskN,
    )
    print(generate_nums, copy_nums)
    train_steps = args.n_epochs * math.ceil(len(train_fold) / args.batch_size)
    output_lang, train_pairs, valid_pairs, test_pairs = prepare_bert_data(
        train_fold, valid_fold, test_fold, generate_nums,
        copy_nums, tokenizer, args.max_seq_length, tree=True)
    print("output vocab:", output_lang.word2index)
    
    print("--------------------------------------------------------------------------------------------------------------------------")
    print("train_valid_test_len:", len(train_pairs), len(valid_pairs), len(test_pairs))
    print("--------------------------------------------------------------------------------------------------------------------------")
    generate_num_ids = []
    for num in generate_nums:
        generate_num_ids.append(output_lang.word2index[num])
    
    # Initialize models
    # encoder = Encoder_Bert(bert_path=args.bert_path)
    encoder = BertModel.from_pretrained(args.bert_path)
    # op_nums in Prediction: ['+', '-', '*', '/']
    retriever = LogicAttn(hidden_size=encoder.config.hidden_size)

    param_optimizer = list(encoder.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay_bert},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    encoder_optimizer = AdamW(optimizer_grouped_parameters,
                    lr = args.learning_rate_bert, # args.learning_rate - default is 5e-5
                    eps = 1e-8, # args.adam_epsilon  - default is 1e-8.
                    correct_bias = False
                    )
    encoder_scheduler = get_linear_schedule_with_warmup(encoder_optimizer, 
                                        num_warmup_steps = int(train_steps * args.warmup_proportion), # Default value in run_glue.py
                                        num_training_steps = train_steps)

    retriever_optimizer = torch.optim.Adam(retriever.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    retriever_scheduler = torch.optim.lr_scheduler.StepLR(retriever_optimizer, step_size=args.step_size, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        retriever.cuda()

    logfile = args.save_path + '/log'
    with open(logfile, 'w') as file_object:
        file_object.write("training procedure log \n")

    train_data = MathWP_Dataset(train_pairs)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    valid_data = MathWP_Dataset(valid_pairs)
    valid_data_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)
    
    test_data = MathWP_Dataset(test_pairs)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate)

    best_R = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0}
    best_P = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0}
    best_F = {'1':0, '2':0, '3':0, '4':0, '5':0, '6':0}
    for epoch in range(args.n_epochs):
        print('epoch:', epoch+1)
        start = time.time()
        random.seed(epoch + args.seed) 
        loss_total = 0

        encoder_optimizer.zero_grad()
        retriever_optimizer.zero_grad()

        with open(logfile, 'a') as file_object:
            file_object.write("epoch: %d \n"%(epoch + 1))

        for batch in tqdm(train_data_loader):
            loss_total += train_retriever(
                batch['inter_multi_label'], 
                encoder, retriever, encoder_optimizer, encoder_scheduler, retriever_optimizer,
                batch["token_ids"], 
                batch["token_type_ids"], 
                batch["attention_mask"],
                logic_token_ids, 
                logic_attention_mask, 
                logic_token_type_ids
            )
        print("loss:", loss_total / len(train_data))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")

        with open(logfile, 'a') as file_object:
            file_object.write("loss: %f\n" %(loss_total / len(train_data)))
            file_object.write("training time " + str(time_since(time.time() - start)) + "\n")
            file_object.write("--------------------------------\n")

        retriever_scheduler.step()

        valid_epoch = 1 #5 if epoch<0.35*args.n_epochs else 2
        if (epoch+1) % valid_epoch == 0 or epoch > args.n_epochs-10:
            encoder.eval()
            retriever.eval()

            print('** train result:')
            R_train, P_train, F1_train, Predict_train = evaluate_result(args, encoder, retriever, train_data_loader, logfile)
            print('** valid result:')
            R_valid, P_valid, F1_valid, Predict_valid = evaluate_result(args, encoder, retriever, valid_data_loader, logfile)
            print('** test result:')
            R_test, P_test, F1_test, Predict_test = evaluate_result(args, encoder, retriever, test_data_loader, logfile)
            
            for key in ['1', '2', '3', '4', '5', '6']:
                if R_valid[key] >= best_R[key]:

                    best_R[key] = R_valid[key]
                    best_P[key] = P_valid[key]
                    best_F[key] = F1_valid[key]

                    torch.save(encoder.state_dict(), "%s/encoder_R" % (args.save_path) + key)
                    torch.save(retriever.state_dict(), "%s/retriever_R" % (args.save_path) + key)

                    with open (args.save_path + "/predicts_train_R"+key, 'wb') as f: #打开文件
                        pickle.dump(Predict_train[key], f)
                    with open (args.save_path + "/predicts_valid_R"+key, 'wb') as f: #打开文件
                        pickle.dump(Predict_valid[key], f)
                    with open (args.save_path + "/predicts_test_R"+key, 'wb') as f: #打开文件
                        pickle.dump(Predict_test[key], f)
                        
                    print('## saving files: R'+key)

            with open(logfile, 'a') as file_object:
                file_object.write(str(best_R)+'\n')
                file_object.write(str(best_P)+'\n')
                file_object.write(str(best_F)+'\n')
        
        print("--------------------------------------------------------------------------------------------------------------------------")
            
            
