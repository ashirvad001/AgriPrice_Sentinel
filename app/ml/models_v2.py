import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        attn_weights = self.attention(lstm_out) # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Context vector: sum of hidden states weighted by attention
        context = torch.sum(attn_weights * lstm_out, dim=1) # [batch_size, hidden_dim]
        return context, attn_weights


class BiLSTM_v2(nn.Module):
    """
    BiLSTM v2 Architecture for Ablation Study.
    Configurable to use Embeddings and/or Temporal Attention.
    """
    def __init__(
        self, 
        num_features: int = 27,
        use_commodity_emb: bool = False,
        use_market_emb: bool = False,
        use_attention: bool = False,
        n_commodities: int = 260,
        n_markets: int = 2518,
        emb_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        self.use_commodity_emb = use_commodity_emb
        self.use_market_emb = use_market_emb
        self.use_attention = use_attention
        
        self.comm_emb = nn.Embedding(n_commodities, emb_dim) if use_commodity_emb else None
        self.mkt_emb = nn.Embedding(n_markets, emb_dim) if use_market_emb else None
        
        # Calculate effective input dimension
        lstm_input_dim = num_features
        if use_commodity_emb: lstm_input_dim += emb_dim
        if use_market_emb:    lstm_input_dim += emb_dim
            
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        lstm_out_dim = hidden_dim * 2
        
        if use_attention:
            self.attention = TemporalAttention(lstm_out_dim)
            
        self.dropout = nn.Dropout(dropout)
        
        # Shared representation before heads
        self.shared_fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Forecasting heads (30d, 60d, 90d)
        self.head_30 = nn.Linear(hidden_dim, 1)
        self.head_60 = nn.Linear(hidden_dim, 1)
        self.head_90 = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq, cat_ids):
        # x_seq: [batch, seq_len, features]
        # cat_ids: [batch, 2] -> (commodity, market)
        
        batch_size, seq_len, _ = x_seq.shape
        
        inputs = [x_seq]
        
        if self.use_commodity_emb:
            c_id = cat_ids[:, 0] # [batch]
            c_emb = self.comm_emb(c_id) # [batch, emb_dim]
            # Expand to sequence length: [batch, seq_len, emb_dim]
            c_emb_seq = c_emb.unsqueeze(1).expand(-1, seq_len, -1)
            inputs.append(c_emb_seq)
            
        if self.use_market_emb:
            m_id = cat_ids[:, 1] # [batch]
            m_emb = self.mkt_emb(m_id) # [batch, emb_dim]
            m_emb_seq = m_emb.unsqueeze(1).expand(-1, seq_len, -1)
            inputs.append(m_emb_seq)
            
        # Concatenate features and embeddings along the feature dimension
        if len(inputs) > 1:
            lstm_in = torch.cat(inputs, dim=2)
        else:
            lstm_in = inputs[0]
            
        # LSTM forward
        out, _ = self.bilstm(lstm_in) # out: [batch, seq_len, hidden_dim*2]
        
        if self.use_attention:
            context, attn_weights = self.attention(out)
        else:
            # Standard: take the last timestep
            context = out[:, -1, :]
            
        context = self.dropout(context)
        
        shared = self.shared_fc(context)
        
        p30 = self.head_30(shared)
        p60 = self.head_60(shared)
        p90 = self.head_90(shared)
        
        # Concatenate outputs: [batch, 3]
        preds = torch.cat([p30, p60, p90], dim=1)
        return preds
