import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class ContextAwareAttention(nn.Module):

    def __init__(self,
                 dim_model: int,
                 dim_context: int,
                 dropout_rate: Optional[float]=0.0):
        super(ContextAwareAttention, self).__init__()
        
        self.dim_model = dim_model
        self.dim_context = dim_context
        self.dropout_rate = dropout_rate
        self.attention_layer = nn.MultiheadAttention(embed_dim=self.dim_model, 
                                                     num_heads=1, 
                                                     dropout=self.dropout_rate, 
                                                     bias=True,
                                                     add_zero_attn=False,
                                                     batch_first=True)


        self.u_k = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_k = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_k = nn.Linear(self.dim_model, 1, bias=False)
        
        self.u_v = nn.Linear(self.dim_context, self.dim_model, bias=False)
        self.w1_v = nn.Linear(self.dim_model, 1, bias=False)
        self.w2_v = nn.Linear(self.dim_model, 1, bias=False)
        

    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor,
                v: torch.Tensor,
                context: Optional[torch.Tensor]=None):
        
        key_context = self.u_k(context)
        value_context = self.u_v(context)

        lambda_k = F.sigmoid(self.w1_k(k) + self.w2_k(key_context))
        lambda_v = F.sigmoid(self.w1_v(v) + self.w2_v(value_context))

        k_cap = (1 - lambda_k) * k + lambda_k * key_context
        v_cap = (1 - lambda_v) * v + lambda_v * value_context

        attention_output, _ = self.attention_layer(query=q,
                                                   key=k_cap,
                                                   value=v_cap)
        return attention_output

class MAF(nn.Module):
    
    def __init__(self,
                 dim_model: int=384,
                 dropout_rate: int=0.2):
        super(MAF, self).__init__()

        ACOUSTIC_DIM = 88
        VISUAL_DIM = 1000
        TEXT_DIM = 384

        self.dropout_rate = dropout_rate

        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias=False)     
        # self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias=False)
        self.acoustic_context_transform = nn.Linear(ACOUSTIC_DIM, TEXT_DIM, bias=False)
        self.visual_context_transform = nn.Linear(VISUAL_DIM, TEXT_DIM, bias=False)

        
        self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                                dim_context=ACOUSTIC_DIM,
                                                                dropout_rate=dropout_rate)
        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                              dim_context=VISUAL_DIM,
                                                              dropout_rate=dropout_rate)        
        self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

        # Output layers
        self.output_complaint = nn.Linear(TEXT_DIM, 2)

        self.output_aspect = nn.Linear(TEXT_DIM, 5)

        # Softmax function
        self.softmax = nn.Softmax(dim=1)
   
        
    # def forward(self,
    #             text_input: torch.Tensor,
    #             acoustic_context: Optional[torch.Tensor]=None,
    #             visual_context: Optional[torch.Tensor]=None):
    def forward(self, input_ids, attention_mask, acoustic_context, visual_context):

        # Text input
        features = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        sbert_output = self.sbert_model(features)
        text_input = sbert_output['sentence_embedding']
        
        # Audio as Context for Attention
        # acoustic_context = acoustic_context.permute(0, 2, 1)
        # acoustic_context = self.acoustic_context_transform(acoustic_context)
        # acoustic_context = acoustic_context.permute(0, 2, 1)
        
        audio_out = self.acoustic_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=acoustic_context)
        
        # Video as Context for Attention
        # visual_context = visual_context.permute(0, 2, 1)
        # visual_context = self.visual_context_transform(visual_context)
        # visual_context = visual_context.permute(0, 2, 1)
        
        video_out = self.visual_context_attention(q=text_input,
                                                  k=text_input,
                                                  v=text_input,
                                                  context=visual_context)
        
        # Global Information Fusion Mechanism
        weight_a = F.sigmoid(self.acoustic_gate(torch.cat((audio_out, text_input), dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat((video_out, text_input), dim=-1)))
        
        output = self.final_layer_norm(text_input +
                                       weight_a * audio_out +
                                       weight_v * video_out)
        
        # Output layer for complaint
        complaint_outputs = self.output_complaint(output)
        complaint_outputs = self.softmax(complaint_outputs)

        # output layer for aspect identification
        aspect_outputs = self.output_aspect(output)
        aspect_outputs = self.softmax(aspect_outputs)

        return complaint_outputs, aspect_outputs