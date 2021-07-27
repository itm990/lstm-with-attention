# LSTMを使用したAttention付き機械翻訳

## 説明

LSTMを使用したAttention付き機械翻訳

## 要件

- Python 3.7.3
- PyTorch 1.6.0
- tqdm 4.56.0
- nltk 3.4.3

## 使用方法

- 学習
```
$ train.py \
    --src_vocab_path [source vocabulary] \
    --tgt_vocab_path [target vocabulary] \
    --src_train_path [source train data] \
    --tgt_train_path [target train data] \
    --src_valid_path [source validation data] \
    --tgt_valid_path [target validation data] \
    --sentence_num 20000 \
    --max_length 50 \
    --batch_size 50 \
    --epoch_size 20 \
    --hidden_size 256 \
    --learning_rate 0.01 \
    --max_norm 5.0 \
    --name [model name] \
    --seed 42 \
    --weight_decay 0.0002
```

- 翻訳
```
$ test.py \
    [model name]/model_state.pt \
    --src_eval_path [source evaluation data] \
    --batch_size 50 \
    --name [output file name]
```