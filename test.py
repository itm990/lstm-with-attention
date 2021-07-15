import torch
import torch.nn as nn
import torch.optim as optim
import configparser
from torch.utils.data import DataLoader
import models
import dataset
import argparse
from tqdm import tqdm



def test(EOS, encoder, decoder, eval_loader, dictionary, max_len, file_name, device):
    
    pbar = tqdm(eval_loader, ascii=True)
    sentences = ''
        
    for batch in pbar:
        
        # source データ
        source = batch.to(device)
        
        # hidden, cell の初期化
        hidden = torch.zeros(source.size(0), encoder.hidden_size, device=device)
        cell   = torch.zeros(source.size(0), encoder.hidden_size, device=device)
        
        
        # ----- encoder へ入力 -----
        
        # encoder の hidden をまとめて decoder に渡す
        source_hiddens = torch.tensor([], device=device)

        # 転置 (batch_size * words_num) --> (words_num * batch_size)
        source_t = torch.t(source)
        
        # source_words は入力単語を表す tensor (batch_size)
        for source_words in source_t:
            hidden, cell = encoder(source_words, hidden, cell)
            source_hiddens = torch.cat((source_hiddens, torch.unsqueeze(hidden, dim=1)), dim=1)
        
        
        # ----- decoder へ入力 -----

        # 最初の入力は <EOS>
        input_words = torch.tensor([EOS] * source.size(0), device=device)

        # 最大文長を max_len + 50 として出力
        batch_words = [[] for _ in range(source.size(0))]
        states = [True for _ in range(source.size(0))]
        for _ in range(max_len + 50):
            output, hidden, cell = decoder(input_words, hidden, cell, source_hiddens, source)
            
            # output (batch_size * dict_size)
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

        # 文章を作成
        for words in batch_words:
            sentences += ' '.join(words) + '\n'

        pbar.set_description('[sentence generation]')

    # 文章をファイルに出力
    with open(file_name, mode='w') as output_f:
        output_f.write(sentences)
                
    print('Fin.')



def main():

    # batch_size の設定
    inifile = configparser.ConfigParser()
    inifile.read('./params_config.ini', 'UTF-8')
    batch_size = int(inifile.get('params', 'batch_size'))
    
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    # データのロード
    url = '/home/ikawa/tutorial/seq2seq/data/ASPEC-JE/corpus.tok/'
    data = torch.load(url+'20000.dict')
    PAD = data['word2index']['src']['<PAD>']
    EOS = data['word2index']['tgt']['<EOS>']
    max_len = data['settings'].max_word_seq_len
    idx2jpn = data['index2word']['tgt']
    eval_data = dataset.SingleDataset(src_data=data['eval']['src'])
    eval_loader = DataLoader(eval_data, batch_size=batch_size, collate_fn=dataset.collate_fn, shuffle=False)

    # モデルのロード
    parser = argparse.ArgumentParser()
    parser.add_argument('model_states_name', type=str)
    args = parser.parse_args()
    model_states_name = args.model_states_name
    print('model:', model_states_name)

    # モデル状態の読込
    model_states = torch.load(model_states_name)
    hidden_size   = model_states['hidden_size']
    src_dict_size = model_states['src_dict_size']
    tgt_dict_size = model_states['tgt_dict_size']
    encoder = models.EncoderLSTM(PAD, hidden_size, src_dict_size).to(device)
    decoder = models.AttentionDecoderLSTM(PAD, hidden_size, tgt_dict_size).to(device)
    encoder.load_state_dict(model_states['encoder_state'])
    decoder.load_state_dict(model_states['decoder_state'])
    
    # 文生成
    test(EOS, encoder, decoder, eval_loader, idx2jpn, max_len, url+'output.attention', device)



if __name__ == '__main__':
    main()
