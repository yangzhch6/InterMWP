# coding: utf-8
import os
import time
import math
import json
import pprint
import argparse
import pickle
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
    # parser.add_argument('--hidden_size', type=int, default=768)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_bert', type=float, default=5e-5)
    parser.add_argument('--weight_decay_bert', type=float, default=1e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--step_size', type=int, default=15)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # 训练控制相关
    parser.add_argument('--test_while_valid', action='store_false', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--maskN', action='store_true', default=False)
    parser.add_argument('--gt_logic', action='store_true', default=False)
    # parser.add_argument('--function', type=str, default="cls")

    # 数据相关参数
    parser.add_argument('--train_data_path', type=str, default="data/train.json")
    parser.add_argument('--valid_data_path', type=str, default="data/valid.json")
    parser.add_argument('--test_data_path' , type=str, default="data/test.json")
    parser.add_argument('--test_full_path' , type=str, default="data/test_full.json")

    parser.add_argument('--logic_path', type=str, default=None)
    parser.add_argument('--prompt_text_path', type=str, default=None)
    parser.add_argument('--retriever_postfix', type=str, default=None)

    # 预训练模型路径
    parser.add_argument('--bert_path', type=str, default="/data1/yangzhicheng/Data/models/chinese-bert-wwm")
    
    # 存储相关参数
    parser.add_argument('--save_path', type=str, default="model/intermwp/baseline")
    parser.add_argument('--save', action='store_true', default=False)

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

if __name__ == "__main__":
    args = set_args()
    #创建save文件夹
    print("** save path:", args.save_path)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("make dir ", args.save_path)

    prompt_text_idx = None
    if args.logic_path != None and args.prompt_text_path!=None:
        prompt_text_idx = dict()
        with open (args.prompt_text_path + "/predicts_train_R"+args.retriever_postfix, 'rb') as f:
            prompt_text_idx['train'] = pickle.load(f)
        with open (args.prompt_text_path + "/predicts_test_R"+args.retriever_postfix, 'rb') as f:
            prompt_text_idx['test'] = pickle.load(f)
        with open (args.prompt_text_path + "/predicts_valid_R"+args.retriever_postfix, 'rb') as f:
            prompt_text_idx['valid'] = pickle.load(f)

    setup_seed(args.seed)
    log_writer = SummaryWriter()

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    train_fold, valid_fold, test_fold, generate_nums, copy_nums = process_data_pipeline(
        args.train_data_path, args.valid_data_path, args.test_data_path, tokenizer,
        debug=args.debug,
        prompt_text_idx=prompt_text_idx,
        logic_path=args.logic_path,
        mask=args.maskN,
        gt=args.gt_logic,
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
    predict = Prediction(hidden_size=encoder.config.hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        input_size=len(generate_nums))
    # op_nums in GenerateNode: ['+', '-', '*', '/', '[NUM]']
    generate = GenerateNode(hidden_size=encoder.config.hidden_size, op_nums=output_lang.n_words - copy_nums - len(generate_nums),
                            embedding_size=args.embedding_size)
    merge = Merge(hidden_size=encoder.config.hidden_size, embedding_size=args.embedding_size)

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


    # encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate_bert, weight_decay=args.weight_decay_bert)
    predict_optimizer = torch.optim.Adam(predict.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generate_optimizer = torch.optim.Adam(generate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    merge_optimizer = torch.optim.Adam(merge.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=25, gamma=0.5)
    predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=args.step_size, gamma=0.5)
    generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=args.step_size, gamma=0.5)
    merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=args.step_size, gamma=0.5)

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        predict.cuda()
        generate.cuda()
        merge.cuda()
    
    best_equ_ac_valid = 0
    best_val_ac_valid = 0

    equation_ac_final = 0
    value_ac_final = 0
    test_total = 0

    logfile = args.save_path + '/log'
    with open(logfile, 'w') as file_object:
        file_object.write("training procedure log \n")

    train_data = MathWP_Dataset(train_pairs)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate)

    train_tree = train_tree_cls
    evaluate_tree = evaluate_tree_cls

    for epoch in range(args.n_epochs):
        start = time.time()
        random.seed(epoch + args.seed) 
        loss_total = 0

        print("epoch:", epoch + 1)
        with open(logfile, 'a') as file_object:
            file_object.write("epoch: %d \n"%(epoch + 1))
        num_accurate = 0

        for batch in tqdm(train_data_loader):
            loss, accurate = train_tree(batch["output"], batch["output_len"], 
                batch["num_size"], generate_nums,
                encoder, predict, generate, merge, encoder_optimizer, encoder_scheduler, 
                predict_optimizer, generate_optimizer,
                merge_optimizer, output_lang, batch["num_idx"], 
                batch["token_ids"], 
                batch["token_type_ids"], 
                batch["attention_mask"])
                
            loss_total += loss
            num_accurate += accurate
        
        predict_scheduler.step()
        generate_scheduler.step()
        merge_scheduler.step()
        
        print("loss:", loss_total / len(train_data), "  accurate:", num_accurate/len(train_pairs))
        print("training time", time_since(time.time() - start))
        print("--------------------------------")

        with open(logfile, 'a') as file_object:
            file_object.write("loss: %f" %(loss_total / len(train_data)) + "  accurate: %f \n"%(num_accurate/len(train_pairs)))
            file_object.write("training time " + str(time_since(time.time() - start)) + "\n")
            file_object.write("--------------------------------\n")

        log_writer.add_scalar('train/loss', loss_total/len(train_data), epoch)
        log_writer.add_scalar('train/accurate', num_accurate/len(train_pairs), epoch)
        
        valid_epoch = 5 if epoch<0.6*args.n_epochs else 2
        if (epoch+1) % valid_epoch == 0 or epoch > args.n_epochs-5:
            value_ac = 0
            equation_ac = 0
            eval_total = 0
            start = time.time()
            """
            data格式:
            {
                "token_ids":token_ids
                "token_type_ids":token_type_ids
                "attention_mask":attention_mask
                "output":output_cell
                "nums":pair["nums"]
                "original_text": text
                "tokens":tokens,
                "num_idx":num_idx,
            }
            """
            for valid_batch in tqdm(valid_pairs):
                token_len = len(valid_batch["tokens"])
                valid_res = evaluate_tree(generate_num_ids, encoder, predict, generate, merge, 
                    output_lang, valid_batch["num_idx"], 
                    [valid_batch["token_ids"][:token_len]], 
                    [valid_batch["token_type_ids"][:token_len]], 
                    [valid_batch["attention_mask"][:token_len]], 
                    token_len, beam_size = args.beam_size)
                val_ac, equ_ac, _, _ = compute_prefix_tree_result(valid_res, valid_batch["output"], 
                    output_lang, valid_batch["nums"])
                if val_ac:
                    value_ac += 1
                if equ_ac:
                    equation_ac += 1
                eval_total += 1
            print(equation_ac, value_ac, eval_total)
            print("valid_answer_acc",  float(value_ac) / eval_total)
            best_equ_ac_valid = max(best_equ_ac_valid, float(equation_ac) / eval_total)
            best_val_ac_valid = max(best_val_ac_valid, float(value_ac) / eval_total)
            print("Best_valid_acc  ", best_val_ac_valid)
            print("validing time", time_since(time.time() - start))
            print("------------------------------------------------------")

            with open(logfile, 'a') as file_object:
                file_object.write("%d %d \n"%(value_ac, eval_total))
                file_object.write("valid_answer_acc: %f \n" %( float(value_ac) / eval_total) )
                file_object.write("Best_valid_acc  : %f \n"% (best_val_ac_valid))
                file_object.write("validing time " + str(time_since(time.time() - start)) + "\n")
                file_object.write("------------------------------------------------------\n")

            is_best_valid = (best_val_ac_valid == float(value_ac)/eval_total)
            if best_val_ac_valid == float(value_ac)/eval_total and args.save:
                # encoder.savebert(args.save_path + "/pytorch_model.bin")
                torch.save(encoder.state_dict(), "%s/encoder" % (args.save_path))
                torch.save(predict.state_dict(), "%s/predict" % (args.save_path))
                torch.save(generate.state_dict(), "%s/generate" % (args.save_path))
                torch.save(merge.state_dict(), "%s/merge" % (args.save_path))

            log_writer.add_scalar('valid/accurate', float(value_ac)/eval_total, epoch)
            
            # _____________________________________________________________________________
            if args.test_while_valid:
                value_ac = 0
                equation_ac = 0
                test_total = 0
                start = time.time()
                """
                data格式:
                {
                    "token_ids":token_ids
                    "token_type_ids":token_type_ids
                    "attention_mask":attention_mask
                    "output":output_cell
                    "nums":pair["nums"]
                    "original_text": text
                    "tokens":tokens,
                    "num_idx":num_idx,
                }
                """
                for test_batch in tqdm(test_pairs):
                    token_len = len(test_batch["tokens"])
                    test_res = evaluate_tree(generate_num_ids, encoder, predict, generate, merge, 
                        output_lang, test_batch["num_idx"], 
                        [test_batch["token_ids"][:token_len]], 
                        [test_batch["token_type_ids"][:token_len]], 
                        [test_batch["attention_mask"][:token_len]], 
                        token_len, beam_size = args.beam_size)
                    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch["output"], 
                        output_lang, test_batch["nums"])
                    if val_ac:
                        value_ac += 1
                    if equ_ac:
                        equation_ac += 1
                    test_total += 1
                print(value_ac, test_total)
                print("test_answer_acc", float(value_ac) / test_total)
                print("testing time", time_since(time.time() - start))
                print("------------------------------------------------------")

                with open(logfile, 'a') as file_object:
                    file_object.write("%d %d \n"%(value_ac, test_total))
                    file_object.write("test_answer_acc: %f \n" %( float(value_ac) / test_total) )
                    file_object.write("testing time " + str(time_since(time.time() - start)) + "\n")
                    file_object.write("------------------------------------------------------\n")

                log_writer.add_scalar('test/accurate', float(value_ac)/test_total, epoch)

                if is_best_valid:
                    equation_ac_final = equation_ac
                    value_ac_final = value_ac
                    
        # 清理显存  
        torch.cuda.empty_cache()

        
    print("__________________________________________________________________________________________")
    print("## Begin Testing ##")
    if args.save:
        # encoder = Encoder_Bert(bert_path = args.save_path)
        encoder.load_state_dict(torch.load(args.save_path + '/encoder'))
        predict.load_state_dict(torch.load(args.save_path + '/predict'))
        generate.load_state_dict(torch.load(args.save_path + '/generate'))
        merge.load_state_dict(torch.load(args.save_path + '/merge'))

        # Move models to GPU
        if USE_CUDA:
            encoder.cuda()
            predict.cuda()
            generate.cuda()
            merge.cuda()
        
        value_ac = 0
        equation_ac = 0
        test_total = 0
        start = time.time()
        predicts = []
        baseline_val_acc = []
        for test_batch in tqdm(test_pairs):
            token_len = len(test_batch["tokens"])
            test_res = evaluate_tree(generate_num_ids, encoder, predict, generate, merge, 
                output_lang, test_batch["num_idx"], 
                [test_batch["token_ids"][:token_len]], 
                [test_batch["token_type_ids"][:token_len]], 
                [test_batch["attention_mask"][:token_len]], 
                token_len, beam_size = args.beam_size)
            predicts.append(test_res)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch["output"], 
                output_lang, test_batch["nums"])
            baseline_val_acc.append(val_ac)
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            test_total += 1
        print(value_ac, test_total)
        print("test_acc", float(value_ac) / test_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")

        with open(logfile, 'a') as file_object:
            file_object.write("Testing...\n")
            file_object.write("%d %d \n"%(value_ac, test_total))
            file_object.write("test_acc: %f \n" %( float(value_ac) / test_total ) )
            file_object.write("------------------------------------------------------\n")
        
        #--------------------------------------------------------------------------
        index2word = [key for key in output_lang.word2index]
        test_full = read_json('data/test_full.json')
        test_str = []
        for line in test_full:
            tmp_strs = []
            for inter in line["interpretation"]:
                tmp_str = []
                find_output_prefix(inter, tmp_str)
                tmp_strs.append(tmp_str)
                # print(tmp_str)
            test_str.append(tmp_strs)
        #--------------------------------------------------------------------------
        test_inter = []
        for line in test_full:
            inter_strs = []
            for inter in line["interpretation"]:
                inter_str = []
                find_inter_prefix(inter, inter_str)
                inter_strs.append(inter_str)
                # print(inter_str)
            test_inter.append(inter_strs)
        # #--------------------------------------------------------------------------
        predicts_str_base = []
        for line in predicts:
            line_str = [index2word[idx] for idx in line]
            predicts_str_base.append(line_str)
            # print(line_str)
        #--------------------------------------------------------------------------
        formula_acc_baseline = []
        for res, ans in zip(predicts_str_base, test_str):
            formula_acc_baseline.append(res in ans)
        print('GTS(Bert) formula acc :', sum(formula_acc_baseline)/len(formula_acc_baseline), sum(formula_acc_baseline), len(formula_acc_baseline))
        
    else:
        print(value_ac_final, test_total)
        print("test_acc", float(value_ac_final) / test_total)
        print("------------------------------------------------------")
        with open(logfile, 'a') as file_object:
            file_object.write("Testing...\n")
            file_object.write("%d %d \n"%(value_ac_final, test_total))
            file_object.write("test_acc: %f \n" %( float(value_ac_final) / test_total ) )
            file_object.write("------------------------------------------------------\n")

