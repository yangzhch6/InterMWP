# 1.Preparing
## 1.0 prepare environment
```
environment.txt
```

torch version
```
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3.1 -c pytorch -c conda-forge
```

torch==3.4.0
## 1.1 Prepare data
first put InterMWP dataset in 'data/'

change PLM's vocab, the top two lines in vocab.text in 'chinese-bert-wwm' should be modified as:
```
[PAD]
[NUM]
...
```

# 2. Training Procedure
## 2.0 Train GTS(Bert) Baseline
run_bert2tree.py    
```
CUDA_VISIBLE_DEVICES=0 python run_bert2tree.py --save_path model/finetune/bert2tree --save
```

**Note: We have already proveid the trained retriever's generated promots, you can skip to 2.2**

## 2.1 train retriever
SoftmaxBCELoss:   
```
CUDA_VISIBLE_DEVICES=0 python run_retriever_attn.py --save_path model/retriever/attn_ep[20] --n_epochs 20
```

## 2.2 prompt-enhanced learning for MWP
top K prompts learning:   
```
CUDA_VISIBLE_DEVICES=2 python run_bert2tree.py --save_path model/finetune/bert2tree_softmacbceloss_ep[20]_top3 --save --logic_path data/logic.json --prompt_text_path model/retriever/SoftmaxBCELoss_ep\[20\]/ --retriever_postfix 3
```

Then generate logic generator training data:   
-   GTS(BERT)
    ```
    CUDA_VISIBLE_DEVICES=5 python evaluate_bert2tree.py --save_path model/finetune/bert2tree/ --save_data_path data/bert2tree --save
    ```
- InterSolver 
    ```
    CUDA_VISIBLE_DEVICES=5 python evaluate_bert2tree.py --save_path model/finetune/bert2tree_softmacbceloss_ep\[20\]_top3 --logic_path data/logic.json --prompt_text_path model/retriever/SoftmaxBCELoss_ep\[20\] --retriever_postfix 3 --save_data_path data/bert2tree_bce_ep20_top3 --save
    ```

## 2.3 train logic generator
GTS(BERT)-[logic generator]
```
CUDA_VISIBLE_DEVICES=5 python run_logic_generator.py --data_path data/bert2tree/ --save_path model/logic_generator/bert2tree/ --save --test_while_valid
```

InterSolver-[logic generator]
```
CUDA_VISIBLE_DEVICES=5 python run_logic_generator.py --data_path data/bert2tree_bce_ep20_top3/ --save_path model/logic_generator/bert2tree_bce_ep20_top3/ --save --test_while_valid
``` 

# 3. prompt learning on Math23K:
## 3.1 Train GTS(Bert) Baseline on Math23K
```
CUDA_VISIBLE_DEVICES=2 python run_bert2tree_math23k.py --save_path model/finetune_math23k/bert2tree --save
```
**Note: We have already proveid the trained retriever's generated promots in Math23K, you can skip to 3.3**

## 3.2 generate Math23K prompts
```
CUDA_VISIBLE_DEVICES=1 python retriever_attn_output.py --save_path model/retriever/SoftmaxBCELoss_ep\[20\] --loss_function SoftmaxBCELoss
```

## 3.3 prompt-enhanced learning in Math23K
```
CUDA_VISIBLE_DEVICES=2 python run_bert2tree_math23k.py --save_path model/finetune_math23k/bert2tree_softmacbceloss_ep[20]_top3 --save --logic_path data/logic.json --retriever_postfix 3
```