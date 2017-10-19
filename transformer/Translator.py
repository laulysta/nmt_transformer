''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
from torch.autograd import Variable

from transformer.Models import Transformer
from transformer.Beam import Beam
from transformer.Models import position_encoding_init
import transformer.Constants as Constants

class Translator(object):
    ''' Load with trained model and handle the beam search '''

    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        checkpoint = torch.load(opt.model, map_location=lambda storage, loc: storage)
        model_opt = checkpoint['settings']
        if 'use_ctx' not in model_opt.__dict__:
            model_opt.use_ctx = False
        self.model_opt = model_opt

        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.tgt_vocab_size,
            model_opt.max_token_seq_len,
            proj_share_weight=model_opt.proj_share_weight,
            embs_share_weight=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner_hid=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout,
            use_ctx=model_opt.use_ctx)

        prob_projection = nn.LogSoftmax()

        model.load_state_dict(checkpoint['model'])

        # New max_token_seq_len for position encoding
        model = self.change_position_embedings(model, opt.max_token_seq_len, model_opt.d_word_vec, model_opt.use_ctx)
        model_opt.max_token_seq_len = opt.max_token_seq_len
        print('[Info] Trained model state loaded.')

        if opt.cuda:
            model.cuda()
            prob_projection.cuda()
        else:
            model.cpu()
            prob_projection.cpu()

        model.prob_projection = prob_projection

        self.model = model
        self.model.eval()

    def change_position_embedings(self, model, max_token_seq_len, d_word_vec, use_ctx):
        n_position = max_token_seq_len + 1
        model.encoder.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        model.encoder.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        model.decoder.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
        model.decoder.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        if use_ctx:
            model.encoder_ctx.position_enc = nn.Embedding(n_position, d_word_vec, padding_idx=Constants.PAD)
            model.encoder_ctx.position_enc.weight.data = position_encoding_init(n_position, d_word_vec)

        return model

    def translate_batch(self, src_batch):
        ''' Translation work in one batch '''

        # Batch size is in different location depending on data.
        if self.model_opt.use_ctx:
            (src_seq, src_pos), (ctx_seq, ctx_pos) = src_batch
        else:
            src_seq, src_pos = src_batch
        batch_size = src_seq.size(0)
        beam_size = self.opt.beam_size

        #- Encode
        enc_outputs, enc_slf_attns = self.model.encoder(src_seq, src_pos)
        enc_output = enc_outputs[-1]

        #--- Repeat data for beam
        src_seq = Variable(src_seq.data.repeat(beam_size, 1))
        enc_output = Variable(enc_output.data.repeat(beam_size, 1, 1))
            

        if self.model_opt.use_ctx:
            #- Encode
            ctx_outputs, ctx_slf_attns = self.model.encoder_ctx(ctx_seq, ctx_pos)
            ctx_output = ctx_outputs[-1]

            #--- Repeat data for beam
            ctx_seq = Variable(ctx_seq.data.repeat(beam_size, 1))
            ctx_output = Variable(ctx_output.data.repeat(beam_size, 1, 1))
                
        #--- Prepare beams
        beam = [Beam(beam_size, self.opt.cuda) for k in range(batch_size)]
        batch_idx = list(range(batch_size))
        n_remaining_sents = batch_size

        #- Decode
        for i in range(self.model_opt.max_token_seq_len):

            len_dec_seq = i + 1
            
            # -- Preparing decode data seq -- #
            input_data = torch.stack([
                b.get_current_state() for b in beam if not b.done]) # size: mb x bm x sq
            
            input_data = input_data.permute(1,0,2).contiguous()
            input_data = input_data.view(-1, len_dec_seq)           # size: (mb*bm) x sq
            input_data = Variable(input_data, volatile=True)

            # -- Preparing decode pos seq -- #
            # size: 1 x seq
            input_pos = torch.arange(1, len_dec_seq + 1).unsqueeze(0)
            # size: (batch * beam) x seq
            input_pos = input_pos.repeat(n_remaining_sents * beam_size, 1)
            input_pos = Variable(input_pos.type(torch.LongTensor), volatile=True)

            if self.opt.cuda:
                input_pos = input_pos.cuda()
                input_data = input_data.cuda()

            # -- Decoding -- #
            if self.model_opt.use_ctx:
                dec_outputs, dec_slf_attns, dec_enc_attns, dec_ctx_attns = self.model.decoder(
                    input_data, input_pos, src_seq, enc_output, ctx_seq, ctx_output)
            else:
                dec_outputs, dec_slf_attns, dec_enc_attns = self.model.decoder(
                    input_data, input_pos, src_seq, enc_output)
            dec_output = dec_outputs[-1][:, -1, :] # (batch * beam) * d_model
            dec_output = self.model.tgt_word_proj(dec_output)
            out = self.model.prob_projection(dec_output)

            # batch x beam x n_words
            word_lk = out.view(beam_size, n_remaining_sents, -1).contiguous()
            
            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[:,idx]):
                    active += [b]

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = self.tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active_enc_info(tensor_var, active_idx):
                ''' Remove the encoder outputs of finished instances in one batch. '''
                batch = tensor_var.data[:n_remaining_sents]
                selected = batch.index_select(0, active_idx)
                data = selected.repeat(beam_size, 1, 1)
                return Variable(data, volatile=True)

            def update_active_seq(seq, active_idx):
                ''' Remove the src sequence of finished instances in one batch. '''
                batch = seq.data[:n_remaining_sents]
                selected = batch.index_select(0, active_idx)
                data = selected.repeat(beam_size, 1)
                return Variable(data, volatile=True)

            src_seq = update_active_seq(src_seq, active_idx)
            enc_output = update_active_enc_info(enc_output, active_idx)

            if self.model_opt.use_ctx:
                ctx_seq = update_active_seq(ctx_seq, active_idx)
                ctx_output = update_active_enc_info(ctx_output, active_idx)
                    
            n_remaining_sents = len(active)

        #- Return useful information
        all_hyp, all_scores = [], []
        n_best = self.opt.n_best

        for b in range(batch_size):
            scores = self.tt.FloatTensor(beam_size+len(beam[b].finish_early_scores)).zero_()
            scores[:beam_size] = beam[b].scores
            for i in range(beam_size, beam_size+len(beam[b].finish_early_scores)):
                scores[i] = beam[b].finish_early_scores[i-beam_size][2]
            beam[b].scores = scores
            scores, ks = beam[b].sort_scores()
            all_scores += [scores[:n_best]]
            hyps = [beam[b].get_hypothesis(k) if k < beam_size
                    else beam[b].get_early_hypothesis(beam[b].finish_early_scores[k-beam_size][0], beam[b].finish_early_scores[k-beam_size][1]) for k in ks[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores
