torch.einsum('bqd,bkd->bqk', query, key)def forward(self, x, mask=None):
    """
    x: Tensor of shape (batch_size, seq_len, embed_dim)
    mask: Optional mask of shape (batch_size, seq_len, seq_len)
    """
    # Step 1: Linear projections
    query = self.query_linear(x)  # Shape: (batch_size, seq_len, embed_dim)
    key = self.key_linear(x)      # Shape: (batch_size, seq_len, embed_dim)
    value = self.value_linear(x)  # Shape: (batch_size, seq_len, embed_dim)
    
    # Step 2: Scaled dot-product attention
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Step 3: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Step 4: Softmax over the last dimension (attention weights)
    attention_weights = F.softmax(scores, dim=-1)
    
    # Step 5: Compute the weighted sum of values
    attention_output = torch.matmul(attention_weights, value)  # Shape: (batch_size, seq_len, embed_dim)
    
    # Step 6: Final linear layer
    output = self.out_linear(attention_output)  # Shape: (batch_size, seq_len, embed_dim)
    
    return output, attention_weights 