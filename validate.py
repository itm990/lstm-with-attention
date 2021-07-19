import torch
import torch.nn as nn
import nltk
from tqdm import tqdm



def validate(BOS, EOS, encoder, decoder, valid_loader, valid_word_data, dictionary, max_len, device):
    
    encoder.eval()
    decoder.eval()
    pbar = tqdm(valid_loader, ascii=True)
    output_sentences = []
        
    for batch in pbar:
        
        # source データ
        source = batch.to(device)
        
        # hidden, cell の初期化
        hidden = torch.zeros(source.size(0), encoder.hidden_size, device=device)
        cell   = torch.zeros(source.size(0), encoder.hidden_size, device=device)
        

        # ----- encoder へ入力 -----
        
        # 転置 (batch_size * words_num) --> (words_num * batch_size)
        source_t = torch.t(source)
        
        # encoder の hidden をまとめて decoder に渡す
        source_hiddens = torch.tensor([], device=device)
        
        # source_words は入力単語を表す tensor (batch_size)
        for source_words in source_t:
            hidden, cell = encoder(source_words, hidden, cell)
            source_hiddens = torch.cat((source_hiddens, torch.unsqueeze(hidden, dim=1)), dim=1)
            

        # ----- decoder へ入力 -----

        # 最初の入力は [BOS]
        input_words = torch.tensor([BOS] * source.size(0), device=device)

        # 最大文長を max_len + 50 として出力
        batch_words = [[] for _ in range(source.size(0))]
        states = [True for _ in range(source.size(0))]
        for _ in range(max_len + 50):
            output, hidden, cell = decoder(input_words, hidden, cell, source_hiddens, source)
            
            # output (batch_size * dict_size) 最大値を取得
            predicted_words = torch.argmax(output, dim=1)
            
            input_words = predicted_words
            
            # batch_words に単語を格納
            for i in range(source.size(0)):
                token_idx = predicted_words[i].item()
                if states[i] == True:
                    if token_idx == EOS:
                        states[i] = False
                    else:
                        batch_words[i].append(dictionary[token_idx])

        for words in batch_words:
            output_sentences.append(words)

        pbar.set_description("[validation]")

    return nltk.translate.bleu_score.corpus_bleu(valid_word_data, output_sentences) * 100
