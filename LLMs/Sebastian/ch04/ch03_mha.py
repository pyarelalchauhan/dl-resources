import torch
import torch.nn as nn
# 3.5 An efficient multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads ==0), \
        "embedded_ dim must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads      #1 . Reduce the projection dimension to match the desired output dim
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)     #2. Uses the linear layer to combine the heads outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length,context_length), diagonal = 1)
        )

    def forward(self,x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x) # 3. tensor shape [b,num_token, d_out)
        quaries = self.W_query(x)
        values = self.W_value(x)

        keys =keys.view(b, num_tokens, self.num_heads, self.head_dim) # 4. We implicitly split the matrix by adding the 
        # num_head dimention. Then we unroll the last dimention (b, num_token, d_out) & (b, num_tokens, num_heads, head_dim)
        quaries = quaries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2) # 5. Transpose from shape (b, num_tokens, num_heads, head_dim ) to (b,num_heads, num_tokens, head_dim)
        queries = quaries.transpose(1,2)
        values = values.transpose(1,2)
        attn_scores =  queries @ keys.transpose(2,3) # 6. Computes dot product for each heads.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] #7. mask truncated the number of tokens
        attn_scores.masked_fill_(mask_bool, -torch.inf)  #8. uses mask to fill attention

        attn_weights = torch.softmax( attn_scores/ keys.shape[-1]**0.5, dim = -1)

        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2) # shape = (b, num_tokens, num_heads, heads_dim)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # Combine heads where _out = self.num_heads * head_dim

        context_vec = self.out_proj(context_vec) # add an optional linear projection 

        return context_vec
        