import torch
import transformers
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.

class BERT_CIUOClass(torch.nn.Module):
    def __init__(self, out_size):
        super(BERT_CIUOClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')
        #self.l1 = transformers.BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.l2 = torch.nn.Dropout(0.05)
        self.l3 = torch.nn.Linear(768, 128)
        self.l4 = torch.nn.LayerNorm((128,), eps=1e-12, elementwise_affine=True)
        self.l5 = torch.nn.Dropout(0.05)
        self.l6 = torch.nn.Linear(128, out_size)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output = self.l6(output_5)
        return output
    
    
class BERT_CAENESClass(torch.nn.Module):
    def __init__(self, out_size):
        super(BERT_CAENESClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es')
        #self.l1 = transformers.BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        self.l2 = torch.nn.Dropout(0.05)
        self.l3 = torch.nn.Linear(768, 256)
        self.l4 = torch.nn.LayerNorm((256,), eps=1e-12, elementwise_affine=True)
        self.l5 = torch.nn.Dropout(0.05)
        self.l6 = torch.nn.Linear(256, out_size)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, return_dict=False)
        output_2 = self.l2(output_1)
        output_3 = self.l3(output_2)
        output_4 = self.l4(output_3)
        output_5 = self.l5(output_4)
        output = self.l6(output_5)
        return output