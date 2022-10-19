# method:
#  utt sample
#  score sample
#  one score per utt
#
# input:
#  SSL feature
#  listener emb
#  text ...
#
#
import torch
import torch.nn as nn
import fairseq
import os
import torch.nn.functional as F

def load_ssl_model(cp_path):
    ssl_model_type = cp_path.split("/")[-1]
    wavlm =  "WavLM" in ssl_model_type
    if wavlm:
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        ssl_model = WavLM(cfg)
        ssl_model.load_state_dict(checkpoint['model'])
        if 'Large' in ssl_model_type:
            SSL_OUT_DIM = 1024
        else:
            SSL_OUT_DIM = 768
    else:
        if ssl_model_type == "wav2vec_small.pt":
            SSL_OUT_DIM = 768
        elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
            SSL_OUT_DIM = 1024
        else:
            print("*** ERROR *** SSL model type " + ssl_model_type + " not supported.")
            exit()
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_path]
        )
        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()
    return SSL_model(ssl_model, SSL_OUT_DIM, wavlm)

class SSL_model(nn.Module):
    def __init__(self,ssl_model,ssl_out_dim,wavlm) -> None:
        super(SSL_model,self).__init__()
        self.ssl_model, self.ssl_out_dim = ssl_model, ssl_out_dim
        self.WavLM = wavlm

    def forward(self,x):
        wav = x
        wav = wav.squeeze(1) # [batches, audio_len]
        if self.WavLM:
            x = self.ssl_model.extract_features(wav)[0]
        else:
            res = self.ssl_model(wav, mask=False, features_only=True)
            x = res["x"]
        return x
    def get_output_dim(self):
        return self.ssl_out_dim


class PhonemeEncoder(nn.Module):
    '''
    PhonemeEncoder consists of an embedding layer, an LSTM layer, and a linear layer.
    Args:
        vocab_size: the size of the vocabulary
        hidden_dim: the size of the hidden state of the LSTM
        emb_dim: the size of the embedding layer
        out_dim: the size of the output of the linear layer
        n_lstm_layers: the number of LSTM layers
    '''
    def __init__(self, vocab_size, hidden_dim, emb_dim, out_dim,n_lstm_layers,with_reference=True) -> None:
        super(PhonemeEncoder, self).__init__()
        print(vocab_size, hidden_dim, emb_dim, out_dim,n_lstm_layers)
        self.with_reference = with_reference
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim,
                               num_layers=n_lstm_layers, dropout=0.1, bidirectional=True)
        self.linear = nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim*self.with_reference, out_dim),
                nn.ReLU()
                )
        self.out_dim = out_dim

    def forward(self, seq, lens, reference_seq, reference_lens):
        emb = self.embedding(seq)
        emb = torch.nn.utils.rnn.pack_padded_sequence(
            emb, lens, batch_first=True, enforce_sorted=False)
        _, (ht, _) = self.encoder(emb)
        feature = ht[-1] + ht[0]
        if self.with_reference:
            if reference_seq==None or reference_lens ==None:
                raise ValueError("reference_batch and reference_lens should not be None when with_reference is True")
            reference_emb = self.embedding(reference_seq)
            reference_emb = torch.nn.utils.rnn.pack_padded_sequence(
                reference_emb, reference_lens, batch_first=True, enforce_sorted=False)
            _, (ht_ref, _) = self.encoder(emb)
            reference_feature = ht_ref[-1] + ht_ref[0]
            feature = self.linear(torch.cat([feature,reference_feature],1))
        else:
            feature = self.linear(feature)
        return feature
    def get_output_dim(self):
        return self.out_dim

class DomainEmbedding(nn.Module):
    def __init__(self,n_domains,domain_dim) -> None:
        super(DomainEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_domains,domain_dim)
        self.output_dim = domain_dim
    def forward(self, x):
        return self.embedding(x.long())
    def get_output_dim(self):
        return self.output_dim

class LDConditioner(nn.Module):
    '''
    Conditions ssl output by listener embedding
    '''
    def __init__(self,input_dim, judge_dim, num_judges=None):
        super(LDConditioner, self).__init__()
        self.input_dim = input_dim
        self.judge_dim = judge_dim
        self.num_judges = num_judges
        assert num_judges != None
        self.judge_embedding = nn.Embedding(num_judges, self.judge_dim)
        # concat [self.output_layer, phoneme features]
        
        self.decoder_rnn = nn.LSTM(
            input_size = self.input_dim + self.judge_dim,
            hidden_size = 512,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        ) # linear?
        self.out_dim = self.decoder_rnn.hidden_size*2

    def get_output_dim(self):
        return self.out_dim

    def forward(self, ssl_feature, domain_feature, phoneme_feature=None, judge_ids=None):
        concatenated_feature = torch.cat((ssl_feature, phoneme_feature.unsqueeze(1).expand(-1, ssl_feature.size(1) ,-1)),dim=2)
        
        concatenated_feature = torch.cat(
            (
                concatenated_feature,
                domain_feature
                .expand(-1, concatenated_feature.size(1), -1),
            ),
            dim=2,
        )
        concatenated_feature = torch.cat(
            (
                concatenated_feature,
                self.judge_embedding(judge_ids)
                .expand(-1, concatenated_feature.size(1), -1),
            ),
            dim=2,
        )
        decoder_output, (h, c) = self.decoder_rnn(concatenated_feature)
        return decoder_output

class Projection(nn.Module):
    def __init__(self, conf, input_dim, hidden_dim, activation, range_clipping=False):
        super(Projection, self).__init__()
        self.range_clipping = range_clipping
        self.hidden_dim = hidden_dim
        if range_clipping:
            self.proj = nn.Tanh()
        self.output_dim = 1
        self.activation = activation
        self.conf = conf
        self.net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 1),
        )
        self.perf_net = nn.Sequential(
            nn.Linear(input_dim * 2, self.hidden_dim),
            self.activation,
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=2)
        )
    
    def forward(self, x):
        if self.conf['pref_mode'] == 'pref':
            pairs = torch.split(x, x.shape[0] // 2, dim=0)
            x = torch.cat([pairs[0], pairs[1]], dim=2)
            output = self.perf_net(x)
            output = F.softmax(output, dim=2)
            output = output.transpose(1, 2)
            # B * 2 * T
            return output

        elif self.conf['pref_mode'] == 'both':
            output = self.perf_net(x)
            output = F.softmax(output, dim=2)
            output = output.transpose(1, 2)

            output1 = self.net(x)
            if self.range_clipping:
                output1 = self.proj(output1) * 2.0 + 3
            return torch.cat([output, output1], dim=2)

        else:
            output = self.net(x)

            # B * T * 1
            # range clipping
            if self.range_clipping:
                return self.proj(output) * 2.0 + 3
            else:
                return output
    def get_output_dim(self):
        return self.output_dim


class Model(torch.nn.Module):
    def __init__(self, conf):
        super(Model, self).__init__()
        self.conf = conf
        self.phoneme_model = PhonemeEncoder(conf['vocab_size'], conf['hidden_dim'], conf['emb_dim'], conf['out_dim'], conf['n_lstm_layers'], with_reference=True)
        self.domain_model = DomainEmbedding(conf['n_domains'],conf['domain_dim'])
        input_dim = 768
        for model in [self.phoneme_model, self.domain_model]:
            input_dim += model.get_output_dim()
        self.ldconditioner = LDConditioner(input_dim, conf['judge_dim'], num_judges=conf['num_judges'])
        input_dim = self.ldconditioner.get_output_dim()
        self.proj = Projection(conf, input_dim, conf['projection_hidden_dim'], torch.nn.ReLU(), range_clipping=False)

    def forward(self, x):
        ssl_feature = x[0].data # input: SSL, B*T*D
        seq = x[1].data # number sequence
        lens = x[1].len 
        reference_seq = x[2].data
        reference_len = x[2].len
        domain = x[3].data  # -> embedding
        judge_id = x[4].data
        try:
            ids = x[5]
            scores = x[6].data
        except IndexError as e:
            pass
        phoneme_feature = self.phoneme_model(seq, lens, reference_seq, reference_len)
        domain_feature = self.domain_model(domain)
        ld_feature = self.ldconditioner(ssl_feature, domain_feature, phoneme_feature, judge_id)
        output = self.proj(ld_feature) 
        if self.conf['pref_mode'] == 'pref':
            split_scores = torch.split(scores, scores.shape[0] // 2, dim=0)
            gt_scores = (split_scores[0] > split_scores[1]).long()
            utt_class_scores = output.mean(dim=2)
            utt_scores = torch.argmax(utt_class_scores, dim=1, keepdim=True)
            ids = [ids[i] + '__&__' + ids[i+len(ids) // 2] for i in range(len(ids) // 2)]
            return output, gt_scores, utt_scores, utt_class_scores, ids
        else:
            utt_scores = output.mean(dim=1)
            return output, utt_scores

    def inference(self, x):
        with torch.no_grad():
            return self(x)