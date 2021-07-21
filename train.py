import argparse
import datetime
import json
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import dataset
from tqdm import tqdm
import models
import validate


def load_sentences(data_path):
    sent_list = []
    with open(data_path) as f:
        for sent in f:
            sent_list.append(sent.strip())
    return sent_list


def convert_sent_to_word(sent_list):
    return [ sent.strip().split(" ") for sent in sent_list ]

def trim_list(src_lst, tgt_lst, sent_num=5000, max_len=100):
    trimmed_src_lst, trimmed_tgt_lst = [], []
    item_cnt = 0
    for src, tgt in zip(src_lst, tgt_lst):
        if item_cnt >= sent_num:
            break
        if len(src) > max_len or len(tgt) > max_len:
            continue
        trimmed_src_lst.append(src)
        trimmed_tgt_lst.append(tgt)
        item_cnt += 1
    return trimmed_src_lst, trimmed_tgt_lst


def convert_word_to_idx(word_list, word2index):
    return [ [ word2index[word] if word in word2index else word2index["[UNK]"] for word in words ] for words in word_list ]


def train(BOS, EOS, encoder, decoder, encoder_optimizer, decoder_optimizer, max_norm, criterion,
          epoch_size, train_loader, valid_loader, valid_word_data, dictionary, max_len, device):

    print("start training.")
    
    for epoch in range(epoch_size):

        encoder.train()
        decoder.train()
        pbar = tqdm(train_loader, ascii=True)
        total_loss = 0

        for i, batch in enumerate(pbar):

            # 初期化
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss = 0

            # データの分割
            enc_in, dec_in, dec_out = map(lambda x: x.to(device), batch)
            
            # hidden, cell の初期化
            hidden = torch.zeros(enc_in.size(0), encoder.hidden_size, device=device)
            cell   = torch.zeros(enc_in.size(0), encoder.hidden_size, device=device)

                        
            # ----- encoder へ入力 -----

            # encoder の hidden をまとめて decoder に渡す
            source_hiddens = torch.tensor([], device=device)

            # 転置 (batch_size * words_num) --> (words_num * batch_size)
            enc_in_t = torch.t(enc_in)

            # input_words は長さ batch_size の 1 次元 tensor
            for source_words in enc_in_t:
                hidden, cell = encoder(source_words, hidden, cell)
                source_hiddens = torch.cat((source_hiddens, torch.unsqueeze(hidden, dim=1)), dim=1)


            # ----- decoder へ入力 -----
            
            # 転置 (batch_size * words_num) --> (words_num * batch_size)
            dec_in_t = torch.t(dec_in)
            dec_out_t = torch.t(dec_out)

            # target_words は長さ batch_size の 1 次元 tensor
            # source_hiddens (batch_size * sentence_length * hidden_size)
            for in_words, out_words in zip(dec_in_t, dec_out_t):
                output, hidden, cell = decoder(in_words, hidden, cell, source_hiddens, enc_in)

                # 損失の計算
                loss += criterion(output, out_words)
                
            total_loss += loss
            loss.backward()

            nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=max_norm)
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=max_norm)
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            pbar.set_description("[epoch:%d] loss:%f" % (epoch+1, total_loss/(i+1)))

        bleu = validate.validate(BOS, EOS, encoder, decoder, valid_loader, valid_word_data, dictionary, max_len, device)
        print("BLEU:", bleu)

    print("Fin.")



def main():
    
    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # パラメータの設定
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--src_dict_path", type=str, default=None)
    parser.add_argument("--tgt_dict_path", type=str, default=None)
    parser.add_argument("--src_train_path", type=str, default=None)
    parser.add_argument("--tgt_train_path", type=str, default=None)
    parser.add_argument("--src_valid_path", type=str, default=None)
    parser.add_argument("--tgt_valid_path", type=str, default=None)
    parser.add_argument("--sentence_num", type=int, default=20000)
    parser.add_argument("--max_length", type=int, default=50)
    
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch_size", type=int, default=20)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_norm", type=float, default=5.0)
    parser.add_argument("--name", type=str, default="no_name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=0.0002)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    # save config file
    save_dir = "./model/{}_{}".format(args.name, datetime_str) if args.name != "no_name" else "./model/no_name"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open("{}/config.json".format(save_dir, mode="w")) as f:
        json.dump(vars(args), f, separators=(",", ":"), indent=4)
    
    # データのロード
    src_dict_data = torch.load(args.src_dict_path)
    tgt_dict_data = torch.load(args.tgt_dict_path)
    src2idx = src_dict_data["dict"]["word2index"]
    tgt2idx = tgt_dict_data["dict"]["word2index"]
    idx2tgt = tgt_dict_data["dict"]["index2word"]
    PAD = src2idx["[PAD]"]
    BOS = src2idx["[BOS]"]
    EOS = src2idx["[EOS]"]
    src_dict_size = len(src2idx)
    tgt_dict_size = len(tgt2idx)
    
    # load train data
    src_train_sent_list = load_sentences(args.src_train_path)
    tgt_train_sent_list = load_sentences(args.tgt_train_path)
    src_valid_sent_list = load_sentences(args.src_valid_path)
    tgt_valid_sent_list = load_sentences(args.tgt_valid_path)
    
    # convert sent to word
    src_train_word_list = convert_sent_to_word(src_train_sent_list)
    tgt_train_word_list = convert_sent_to_word(tgt_train_sent_list)
    src_valid_word_list = convert_sent_to_word(src_valid_sent_list)
    tgt_valid_word_list = convert_sent_to_word(tgt_valid_sent_list)
    
    # trim word list
    src_train_word_list, tgt_train_word_list = trim_list(
        src_train_word_list, tgt_train_word_list, sent_num=args.sentence_num, max_len=args.max_length
    )
    
    # convert word to idx
    src_train_idx_list = convert_word_to_idx(word_list=src_train_word_list, word2index=src2idx)
    tgt_train_idx_list = convert_word_to_idx(word_list=tgt_train_word_list, word2index=tgt2idx)
    src_valid_idx_list = convert_word_to_idx(word_list=src_valid_word_list, word2index=src2idx)
    
    train_data = dataset.PairedDataset(bos_idx=BOS, eos_idx=EOS, src_data=src_train_idx_list, tgt_data=tgt_train_idx_list)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=dataset.paired_collate_fn, shuffle=True)
    valid_data = dataset.SingleDataset(src_data=src_valid_idx_list)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    valid_word_data = [ [words] for words in tgt_valid_word_list ]
    
    # 設定
    encoder = models.EncoderLSTM(PAD, args.hidden_size, src_dict_size).to(device)
    decoder = models.AttentionDecoderLSTM(PAD, args.hidden_size, tgt_dict_size).to(device)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss(ignore_index=PAD, reduction="sum")
    
    # 学習
    train(BOS, EOS, encoder, decoder, encoder_optimizer, decoder_optimizer, args.max_norm, criterion,
          args.epoch_size, train_loader, valid_loader, valid_word_data, idx2tgt, args.max_length, device)

    # モデル状態の保存
    model_states = {
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict()
    }
    torch.save(model_states, "{}/model_state.pt".format(save_dir))
    print("model_name:", args.name)



if __name__ == "__main__":
    main()
