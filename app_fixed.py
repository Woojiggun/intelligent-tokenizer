#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent Tokenizer - Hugging Face Space Demo (Fixed Version)
"""

import gradio as gr
import torch
import torch.nn as nn
import time
from huggingface_hub import hf_hub_download

# ByteTokenizer class
class ByteTokenizerV6:
    def __init__(self, max_seq_len=256):
        self.vocab_size = 260
        self.max_seq_len = max_seq_len
        self.PAD = 256
        self.BOS = 257
        self.EOS = 258  
        self.MASK = 259
    
    def encode(self, text):
        """Convert text to byte sequence"""
        byte_seq = list(text.encode('utf-8'))
        
        # Split into chunks (256 byte limit)
        if len(byte_seq) > self.max_seq_len - 2:
            chunks = []
            for i in range(0, len(byte_seq), self.max_seq_len - 2):
                chunk = byte_seq[i:i + self.max_seq_len - 2]
                chunks.append([self.BOS] + chunk + [self.EOS])
            return chunks
        else:
            return [[self.BOS] + byte_seq + [self.EOS]]
    
    def decode(self, ids):
        """Convert byte sequence to text"""
        # Filter out special tokens and invalid IDs
        filtered = []
        for id in ids:
            if isinstance(id, int) and 0 <= id < 256:
                filtered.append(id)
        
        if not filtered:
            return "[Empty result]"
            
        try:
            return bytes(filtered).decode('utf-8', errors='replace')
        except Exception as e:
            return f"[Decoding error: {str(e)}]"

# Model architecture
class BoundaryAwareTokenizerModel(nn.Module):
    def __init__(self, vocab_size=260, hidden_size=768, num_encoder_layers=5, 
                 num_decoder_layers=6, num_heads=8, dropout=0.1, max_position_embeddings=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=256)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, tgt=None):
        # Encoder
        src_embeddings = self.embedding(src)
        seq_len = src.shape[1]
        positions = torch.arange(seq_len, device=src.device).unsqueeze(0).expand_as(src)
        src_embeddings = src_embeddings + self.position_embedding(positions)
        src_embeddings = self.dropout(src_embeddings)
        
        encoder_output = self.encoder(src_embeddings)
        
        if tgt is not None:
            # Decoder
            tgt_embeddings = self.embedding(tgt)
            tgt_len = tgt.shape[1]
            tgt_positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand_as(tgt)
            tgt_embeddings = tgt_embeddings + self.position_embedding(tgt_positions)
            tgt_embeddings = self.dropout(tgt_embeddings)
            
            # Create causal mask
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
            
            decoder_output = self.decoder(tgt_embeddings, encoder_output, tgt_mask=tgt_mask)
            logits = self.output_projection(decoder_output)
            
            return logits
        else:
            return encoder_output

# Load model
device = torch.device('cpu')  # HF Space uses CPU by default
model = None
tokenizer = ByteTokenizerV6()

def load_model():
    global model
    try:
        # Download model from Hugging Face
        model_file = hf_hub_download(