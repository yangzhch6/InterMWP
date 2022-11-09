# InterMWP Dataset

## files in 'data/'
```
1. train.json, valid.json, and test.json are the train, valid, and test split of InterMWP dataset.
2. test_full.json: We generate equivalent
solution equations as many as possible for each
MWP in the test set so that we can measure the
ability of an MWP solver better.
3. logic.json: All the logic formulas that we've summed up.
```

## data sample
```
{
    "original_text": "同学们去公园春游，四年级去了269人，五年级去了232人，六年级去了168人．三个年级一共去了多少人？",
    "mask_text": "同学 们 去 公园 春游 ， 四年级 去 了 N0 人 ， 五年级 去 了 N1 人 ， 六年级 去 了 N2 人 ． 三个 年级 一共 去 了 多少 人 ？",
    "output_prefix": "+ + N0 N1 N2",
    "output_infix": "N0+N1+N2",
    "output_original": "x=269+232+168",
    "nums": "269 232 168",
    "id": "a9a24f26be8c11eba8c004d4c4250d10",
    "exam_point": "整数的加法和减法",
    "interpretation": {
        "logic": 0,
        "op": "+",
        "left": {
            "logic": 0,
            "op": "+",
            "left": {
                "logic": -1,
                "op": "N0",
                "left": {},
                "right": {}
            },
            "right": {
                "logic": -1,
                "op": "N1",
                "left": {},
                "right": {}
            }
        },
        "right": {
            "logic": -1,
            "op": "N2",
            "left": {},
            "right": {}
        }
    }
},
```