# 1.Preparing
## 1.0 prepare environment
```
environment.txt
```

torch version
```
conda install pytorch==1.10.2 torchvision==0.11.3 torchaudio==0.10.2 cudatoolkit=11.3.1 -c pytorch -c conda-forge
```

transformers
```
pip install transformers==3.4.0
```

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
CUDA_VISIBLE_DEVICES=0 python run_retriever_attn.py --save_path model/retriever/attn_ep[30] --n_epochs 30
```

## 2.2 prompt-enhanced learning for MWP
top K prompts learning:   
```
CUDA_VISIBLE_DEVICES=2 python run_bert2tree.py --save_path model/finetune/bert2tree_softmacbceloss_ep[20]_top3 --save --logic_path data/logic.json --prompt_text_path model/retriever/attn_ep[30] --retriever_postfix 3
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
