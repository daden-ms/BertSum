"""
Portions Copyright (c) Microsoft Corporation
"""
import torch
import torch.nn as nn
#from pytorch_pretrained_bert import BertModel, BertConfig
from transformers import BertModel, BertConfig, PreTrainedModel,  PretrainedConfig
from torch.nn.init import xavier_uniform_

from bertsum.models.encoder import TransformerInterEncoder, Classifier, RNNEncoder
from bertsum.models.optimizers import Optimizer


def build_optim(optimization_method, lr, max_grad_norm, beta1, beta2, decay_method, warmup_steps, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if checkpoint:
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            optimization_method, lr, max_grad_norm,
            beta1=beta1, beta2=beta2,
            decay_method=decay_method,
            warmup_steps=warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    if checkpoint:
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim


class Transformer(nn.Module):
    def __init__(self, temp_dir, model_class, pretrained_model_name, pretrained_config):
        super(Transformer, self).__init__()
        if(pretrained_model_name):
            self.model = model_class.from_pretrained(pretrained_model_name,
                                                   cache_dir=temp_dir)
            #self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)
        else:
            self.model = model_class(pretrained_config)

    def forward(self, x, segs, mask):
        if "DistilBertModel" in str(type(self.model)):
            outputs = self.model(x, attention_mask =mask)
        else:
            outputs = self.model(x, token_type_ids=segs, attention_mask =mask)
        #print(outputs)
        #print(len(outputs))
        top_vec = outputs[0] 
        
        return top_vec



class Summarizer(nn.Module):
    def __init__(self, encoder, args, model_class, pretrained_model_name, pretrained_config = None, temp_dir="./"):
        super(Summarizer, self).__init__()
        self.loss = torch.nn.BCELoss(reduction='none')
        #self.device = device
        self.transformer = Transformer(temp_dir, model_class, pretrained_model_name, pretrained_config)
        if (encoder == 'classifier'):
            self.encoder = Classifier(self.transformer.model.config.hidden_size)
        elif(encoder=='transformer'):
            self.encoder = TransformerInterEncoder(self.transformer.model.config.hidden_size, args.ff_size, args.heads,
                                                   args.dropout, args.inter_layers)
        elif(encoder=='rnn'):
            self.encoder = RNNEncoder(bidirectional=True, num_layers=1,
                                      input_size=self.transformer.model.config.hidden_size, hidden_size=args.rnn_size,
                                      dropout=args.dropout)
        elif (encoder == 'baseline'):
            bert_config = BertConfig(self.transformer.model.config.vocab_size, hidden_size=args.hidden_size,
                                     num_hidden_layers=6, num_attention_heads=8, intermediate_size=args.ff_size)
            self.transformer.model = BertModel(bert_config)
            self.encoder = Classifier(self.transformer.model.config.hidden_size)

        if args.param_init != 0.0:
            for p in self.encoder.parameters():
                p.data.uniform_(-args.param_init, args.param_init)
        if args.param_init_glorot:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        #self.to(device)
    def load_cp(self, pt):
        self.load_state_dict(pt['model'], strict=True)

    def forward(self, x, segs, clss, mask, mask_cls, labels=None, sentence_range=None):

        top_vec = self.transformer(x, segs, mask)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)
        if labels is not None:
            loss = self.loss(sent_scores, labels.float())
            loss = (loss*mask_cls.float()).sum()
            sent_scores = sent_scores + mask_cls.float()
            return loss, sent_scores, mask_cls
        else:
            sent_scores = sent_scores + mask_cls.float()
            return sent_scores, mask_cls
