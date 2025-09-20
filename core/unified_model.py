"""
Unified Intelligent Tokenizer Model v6.1
순수 학습 기반 with Curriculum Learning
- Layer 0: Curriculum Learning for boundaries
- Layer 1: Autonomous language pattern discovery (no labels)
- Layer 2-4: Group-aware relative position encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple, Union


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Transformer 원본 방식)
    학습 가능한 위치 임베딩 대신 고정된 sin/cos 패턴 사용
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create sinusoidal position encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions
        
        # Register as buffer (not trainable)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ByteTokenizer:
    """
    Pure byte-level tokenizer - no language rules
    """
    
    def __init__(self, max_seq_len: int = 256):  # v6.1: 256 bytes for streaming approach
        self.max_seq_len = max_seq_len
        self.PAD = 256
        self.BOS = 257
        self.EOS = 258
        self.MASK = 259
    
    def encode(self, text: str, add_special_tokens: bool = True) -> Dict[str, torch.Tensor]:
        # Convert to UTF-8 bytes
        byte_seq = list(text.encode('utf-8'))
        
        # Truncate if needed
        if len(byte_seq) > self.max_seq_len - 2:
            byte_seq = byte_seq[:self.max_seq_len - 2]
        
        # Add special tokens
        if add_special_tokens:
            byte_seq = [self.BOS] + byte_seq + [self.EOS]
        
        input_ids = torch.tensor(byte_seq, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'length': len(input_ids)
        }
    
    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        encoded = [self.encode(text) for text in texts]
        max_len = min(max(e['length'] for e in encoded), self.max_seq_len)
        
        batch_size = len(texts)
        input_ids = torch.full((batch_size, max_len), self.PAD, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
        
        for i, enc in enumerate(encoded):
            seq_len = min(enc['length'], max_len)
            input_ids[i, :seq_len] = enc['input_ids'][:seq_len]
            attention_mask[i, :seq_len] = 1.0
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, input_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy().tolist()
        
        if skip_special_tokens:
            input_ids = [b for b in input_ids if b < 256]
        
        try:
            byte_array = bytes([min(b, 255) for b in input_ids if b != self.PAD])
            return byte_array.decode('utf-8', errors='replace')
        except:
            return "".join([chr(b) if b < 128 else '?' for b in input_ids if b < 256])


class ByteEncoderV61(nn.Module):
    """
    v6.1: 5-Layer Encoder with Layer-Specialized Architecture
    Layer 0: 768d - Byte to character (with curriculum learning)
    Layer 1: 896d - Language pattern discovery (no labels)
    Layer 2: 1024d - Eojeol/Word formation (+ eojeol PE)
    Layer 3: 1152d - Small phrase grouping (2-3 eojeols)
    Layer 4: 1280d - Final refinement (+ context PE)

    Target: 어절(eojeol) to 구(phrase) level compression (3:1 ratio)
    """

    def __init__(
        self,
        vocab_size: int = 260,
        hidden_dims: List[int] = [768, 896, 1024, 1152, 1280],  # v6.1 dimensions
        num_heads: List[int] = [12, 14, 16, 18, 20],  # v6.1: Progressive heads per layer
        dropout: float = 0.1,
        max_seq_len: int = 256  # v6.1: 256 chunk for streaming
    ):
        super().__init__()
        
        # Layer 0: Byte to Character with Curriculum Learning
        self.byte_embedding = nn.Embedding(vocab_size, hidden_dims[0])

        # v6.1: Multi-level boundary predictors for hierarchical segmentation
        # Level 1: Character boundaries (UTF-8 multi-byte)
        self.char_boundary_predictor = nn.Linear(hidden_dims[0], 3)  # 0: continue, 1: start, 2: end

        # Level 2: Eojeol boundaries (space + particle analysis)
        self.eojeol_boundary_predictor = nn.Linear(hidden_dims[2], 4)  # 0: inside, 1: space, 2: particle, 3: punct

        # Level 3: Phrase boundaries (syntactic chunks)
        self.phrase_boundary_predictor = nn.Linear(hidden_dims[3], 3)  # 0: inside, 1: weak boundary, 2: strong boundary

        # v6.1: Positional encoding ONLY for Layer 0
        self.pos_encoding = PositionalEncoding(hidden_dims[0], max_seq_len, dropout)

        # v6.1: Layer 1 - Language pattern discovery (no labels!)
        self.pattern_discoverer = nn.Linear(hidden_dims[1], 256)  # Discover patterns autonomously (from 896d)
        self.lang_signal_generator = nn.Linear(hidden_dims[1], 128)  # Generate language signals (from 896d)

        # v6.1: Group-aware relative position encodings for Layer 2-4
        self.group_pe_layer2 = nn.Embedding(max_seq_len, hidden_dims[2])  # For eojeol/word units
        self.group_pe_layer3 = nn.Embedding(max_seq_len, hidden_dims[3])  # For small phrases (2-3 eojeols)
        self.group_pe_layer4 = nn.Embedding(max_seq_len, hidden_dims[4])  # For context/discourse

        # 5 Transformer layers with dimension changes
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            input_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            output_dim = hidden_dims[i]

            # Projection layer if dimension changes
            if input_dim != output_dim:
                proj = nn.Linear(input_dim, output_dim)
            else:
                proj = None

            # v6.1: Layer-specific head count for optimal dimension per head
            # Target: 64-80 dim per head
            layer_heads = num_heads[i] if isinstance(num_heads, list) else num_heads

            # Transformer encoder layer
            layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=layer_heads,
                dim_feedforward=output_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            
            self.layers.append(nn.ModuleDict({
                'projection': proj,
                'transformer': layer,
                'norm': nn.LayerNorm(output_dim)
            }))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        boundary_labels: Optional[torch.Tensor] = None,
        epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        v6.1 Forward pass with curriculum learning
        Args:
            boundary_labels: UTF-8 boundary labels for curriculum learning (training only)
            epoch: Current epoch for curriculum schedule
        """
        batch_size, seq_len = input_ids.shape

        # Layer 0: Byte embedding with curriculum learning
        x = self.byte_embedding(input_ids)

        # v6.1: Positional encoding ONLY at Layer 0
        x = self.pos_encoding(x)

        # v6.1: Predict character boundaries (Layer 0)
        char_boundaries = self.char_boundary_predictor(x)

        # v6.1: Curriculum learning for character boundaries
        if self.training and boundary_labels is not None:
            # Calculate curriculum ratio
            if epoch < 30:
                usage_ratio = 1.0  # 100% label usage
            elif epoch < 70:
                usage_ratio = 1.0 - (epoch - 30) / 40  # Gradual decrease
            else:
                usage_ratio = 0.0  # Pure prediction

            if usage_ratio > 0:
                # Mix predicted with true boundaries
                boundary_probs = F.softmax(char_boundaries, dim=-1)
                true_boundary_one_hot = F.one_hot(boundary_labels, num_classes=3).float()
                mixed_boundaries = usage_ratio * true_boundary_one_hot + (1 - usage_ratio) * boundary_probs
                char_boundary_weights = mixed_boundaries
            else:
                char_boundary_weights = F.softmax(char_boundaries, dim=-1)
        else:
            # Inference: pure prediction
            char_boundary_weights = F.softmax(char_boundaries, dim=-1)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Keep attention mask as is for TransformerEncoderLayer
            # It expects shape (batch_size, seq_len) and handles masking internally
            pass
        
        # v6.1: Process through 5 specialized layers
        all_hidden_states = []
        discovered_patterns = None
        eojeol_boundaries = None
        phrase_boundaries = None

        for i, layer_dict in enumerate(self.layers):
            # Project if needed (before layer-specific processing)
            if layer_dict['projection'] is not None:
                x = layer_dict['projection'](x)

            # Layer 1: Add language signals (autonomous discovery)
            if i == 1:
                # Discover language patterns WITHOUT labels (x is now 896d)
                discovered_patterns = self.pattern_discoverer(x)
                lang_signals = self.lang_signal_generator(x)

            # Layer 2: Predict eojeol boundaries and add position encoding
            elif i == 2:
                # Predict eojeol boundaries (spaces, particles, punctuation)
                eojeol_boundaries = self.eojeol_boundary_predictor(x)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                group_pe = self.group_pe_layer2(positions)
                x = x + group_pe * 0.1  # Mild addition to preserve main signal

            # Layer 3: Predict phrase boundaries and add position encoding
            elif i == 3:
                # Predict phrase boundaries (weak/strong syntactic breaks)
                phrase_boundaries = self.phrase_boundary_predictor(x)
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                group_pe = self.group_pe_layer3(positions)
                x = x + group_pe * 0.1

            elif i == 4:
                positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                group_pe = self.group_pe_layer4(positions)
                x = x + group_pe * 0.1

            # Transformer layer - properly handle mask
            if attention_mask is not None:
                key_padding_mask = (attention_mask == 0)
                x = layer_dict['transformer'](x, src_key_padding_mask=key_padding_mask)
            else:
                x = layer_dict['transformer'](x)
            x = layer_dict['norm'](x)
            all_hidden_states.append(x)
        
        # Pool for sequence representation
        if attention_mask is not None:
            # Masked mean pooling - attention_mask is (batch, seq)
            mask = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        
        return {
            'last_hidden_state': x,
            'pooled_output': pooled,
            'all_hidden_states': all_hidden_states,
            # v6.1 boundary predictions
            'char_boundaries': char_boundaries,
            'char_boundary_weights': char_boundary_weights,
            'eojeol_boundaries': eojeol_boundaries,
            'phrase_boundaries': phrase_boundaries,
            'discovered_patterns': discovered_patterns
        }


class CrossAttention(nn.Module):
    """
    Enhanced Cross-attention for relation learning between sequences
    추론 레이어 연결을 위한 강화된 관계 학습
    """
    
    def __init__(self, hidden_dim: int = 1280, num_heads: int = 20, dropout: float = 0.1):
        super().__init__()

        # v6.1: Adjusted for 1280d (64 per head with 20 heads)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True
        )
        
        # v6.1: Enhanced relation classifier with reconstruction focus
        # 0: identity (완벽한 복원), 1: similar, 2: different, 3: continuation
        # 4: translation, 5: summary, 6: expansion, 7: contradiction
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 8)
        )

        # v6.1: Reconstruction-specific attention (복원 전용 어텐션)
        # Use 10 heads for reconstruction (128 per head)
        self.reconstruction_attn = nn.MultiheadAttention(
            hidden_dim, 10, dropout * 0.5, batch_first=True
        )
        
        # Gating mechanism for adaptive fusion
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        key_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Normalize inputs
        query_norm = self.norm1(query)
        key_norm = self.norm2(key)
        
        # Fix key_mask dimension if needed
        if key_mask is not None:
            # Ensure key_mask matches key sequence length
            if key_mask.dim() == 2 and key_mask.size(1) != key.size(1):
                # Create new mask with correct dimensions
                batch_size = key.size(0)
                seq_len = key.size(1)
                key_mask = torch.ones(batch_size, seq_len, dtype=key_mask.dtype, device=key_mask.device)
        
        # Cross attention
        attn_output, attn_weights = self.cross_attn(
            query_norm, key_norm, key_norm,
            key_padding_mask=(key_mask == 0) if key_mask is not None else None
        )
        
        # Residual connection
        attn_output = attn_output + query

        # v6.1: Reconstruction-focused attention (복원 최적화)
        recon_output, recon_weights = self.reconstruction_attn(
            query_norm, query_norm, query_norm,  # Self-attention for consistency
            key_padding_mask=(query_mask == 0) if query_mask is not None else None
        )

        # Combine cross and reconstruction attention
        combined_attn = attn_output * 0.7 + recon_output * 0.3

        # Adaptive gating for fusion
        gate_input = torch.cat([query.mean(dim=1), key.mean(dim=1)], dim=-1)
        gate_weights = self.gate(gate_input).unsqueeze(1)

        # Gated fusion: 적응적으로 attention 결과 조절
        fused_output = gate_weights * combined_attn + (1 - gate_weights) * query
        
        # Pool for relation classification
        query_pooled = query.mean(dim=1) if query_mask is None else \
                      (query * query_mask.unsqueeze(-1)).sum(1) / query_mask.sum(1, keepdim=True).clamp(min=1e-9)
        key_pooled = key.mean(dim=1) if key_mask is None else \
                    (key * key_mask.unsqueeze(-1)).sum(1) / key_mask.sum(1, keepdim=True).clamp(min=1e-9)
        
        # Classify relations with enhanced head
        combined = torch.cat([query_pooled, key_pooled], dim=-1)
        relation_logits = self.relation_head(combined)
        
        return {
            'cross_attention': fused_output,  # Gated fusion output
            'attention_weights': attn_weights,
            'reconstruction_weights': recon_weights,  # v6.1: 복원 어텐션 가중치
            'relation_logits': relation_logits,
            'gate_weights': gate_weights.squeeze(1),  # For analysis
            'reconstruction_score': F.softmax(relation_logits, dim=-1)[:, 0]  # identity 확률 (복원도)
        }


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder with Positional Encoding
    """

    def __init__(
        self,
        vocab_size: int = 260,
        hidden_dim: int = 1280,  # v6.1: Match final encoder dim
        num_heads: int = 16,      # v6.1: 1280/16 = 80 per head
        num_layers: int = 8,      # v6.1 FINAL: 8 layers for better reconstruction
        dropout: float = 0.1,
        max_seq_len: int = 256    # v6.1: 256 chunk for streaming
    ):
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        decoder_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = encoder_hidden.size(0)
        
        # Start with BOS if no input
        if decoder_input_ids is None:
            decoder_input_ids = torch.full((batch_size, 1), 257, device=encoder_hidden.device)
        
        # Embed and add positional encoding
        dec_seq_len = decoder_input_ids.size(1)
        x = self.token_embedding(decoder_input_ids)
        x = self.pos_encoding(x)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(dec_seq_len, dec_seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )
        
        # Decoder forward - handle variable-length encoder outputs
        # The encoder may compress the sequence, so memory (encoder_hidden) might be shorter
        # than the decoder sequence. This is expected and correct behavior.
        enc_seq_len = encoder_hidden.size(1)
        
        # Adjust encoder mask if needed
        if encoder_mask is not None:
            if encoder_mask.size(1) != enc_seq_len:
                # Encoder compressed the sequence, create new mask for compressed length
                # All compressed positions are valid (not masked)
                memory_key_padding_mask = torch.zeros(
                    encoder_hidden.size(0), enc_seq_len,
                    dtype=torch.bool, device=encoder_hidden.device
                )
            else:
                memory_key_padding_mask = (encoder_mask == 0)
        else:
            memory_key_padding_mask = None
            
        # Decoder attends to compressed encoder states via cross-attention
        # This naturally handles different sequence lengths
        decoder_output = self.transformer(
            tgt=x,  # Decoder sequence (original length)
            memory=encoder_hidden,  # Encoder sequence (possibly compressed)
            tgt_mask=causal_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=(decoder_mask == 0) if decoder_mask is not None else None
        )
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return {
            'logits': logits,
            'hidden_states': decoder_output
        }
    
    @torch.no_grad()
    def generate(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        temperature: float = 0.1,  # 토크나이저는 보수적 생성 (정확한 복원)
        top_k: int = 10,  # 상위 10개만 고려
        top_p: float = 0.95
    ) -> torch.Tensor:
        batch_size = encoder_hidden.size(0)
        device = encoder_hidden.device
        
        # Start with BOS
        decoder_input_ids = torch.full((batch_size, 1), 257, device=device)
        
        for _ in range(max_length - 1):
            # Forward pass
            outputs = self.forward(encoder_hidden, decoder_input_ids, encoder_mask)
            next_token_logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)

            # Stop at EOS - check each sequence individually
            if (next_tokens == 258).any():  # EOS token - stop if ANY sequence hits EOS
                break
        
        return decoder_input_ids


class IntelligentTokenizerModelV61(nn.Module):
    """
    Complete Intelligent Tokenizer Model v6.1
    Pure learning-based with curriculum learning
    - No language labels during training
    - Curriculum learning for boundaries
    - Group-aware position encodings
    """

    def __init__(
        self,
        vocab_size: int = 260,
        encoder_dims: List[int] = [768, 896, 1024, 1152, 1280],  # v6.1 dimensions
        encoder_heads: List[int] = [12, 14, 16, 18, 20],  # v6.1: Optimal heads per layer
        decoder_hidden: int = 1280,  # Match final encoder dim
        decoder_heads: int = 16,     # v6.1: 80 per head for decoder
        num_decoder_layers: int = 8,  # v6.1 FINAL: 8 layers for better reconstruction
        dropout: float = 0.1,
        max_seq_len: int = 256  # v6.1: 256 chunk for streaming
    ):
        super().__init__()

        # v6.1 Components with optimized head counts
        self.tokenizer = ByteTokenizer(max_seq_len)
        self.encoder = ByteEncoderV61(vocab_size, encoder_dims, encoder_heads, dropout, max_seq_len)
        self.decoder = TransformerDecoder(vocab_size, decoder_hidden, decoder_heads, num_decoder_layers, dropout, max_seq_len)
        self.cross_attention = CrossAttention(encoder_dims[-1], 20, dropout)  # 20 heads for 1280d
        
    def forward(
        self,
        input_texts: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        boundary_labels: Optional[torch.Tensor] = None,  # v6.1: for curriculum learning
        epoch: int = 0,  # v6.1: for curriculum schedule
        use_cross_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Tokenize if text input
        if input_texts is not None:
            tokenized = self.tokenizer.encode_batch(input_texts)
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
        
        # 시퀀스 길이 체크 및 조정
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # v6.1: Encode with curriculum learning
        encoder_outputs = self.encoder(input_ids, attention_mask, boundary_labels, epoch)
        encoder_hidden = encoder_outputs['last_hidden_state']  # v6.1: [batch, seq, 1280]

        # v6.1: 차원 확인 - 최종 차원은 1280
        assert encoder_hidden.size(-1) == 1280, f"Encoder dim mismatch: {encoder_hidden.size(-1)}"

        # Prepare decoder input for teacher forcing during training
        if decoder_input_ids is None and labels is not None:
            # During training, use shifted labels as decoder input (teacher forcing)
            # Add BOS at the beginning and remove last token
            bos_tokens = torch.full((batch_size, 1), self.tokenizer.BOS, device=labels.device, dtype=labels.dtype)
            decoder_input_ids = torch.cat([bos_tokens, labels[:, :-1]], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            encoder_hidden,
            decoder_input_ids,
            attention_mask
        )
        decoder_hidden = decoder_outputs['hidden_states']  # [batch, seq, 768]
        
        # Cross-Attention (마지막 레이어에서 관계 학습)
        cross_attn_outputs = None
        relation_logits = None
        
        if use_cross_attention and decoder_hidden is not None:
            # 디코더 출력과 인코더 출력 간 크로스어텐션
            cross_attn_outputs = self.cross_attention(
                query=decoder_hidden,  # 디코더가 query
                key=encoder_hidden,     # 인코더가 key/value
                query_mask=None,        # decoder mask는 causal이므로 별도 처리
                key_mask=attention_mask
            )
            
            # 관계 학습 결과
            relation_logits = cross_attn_outputs['relation_logits']
            
            # Cross-attention으로 강화된 디코더 표현
            enhanced_decoder = decoder_hidden + cross_attn_outputs['cross_attention']
            
            # 최종 로짓 재계산 (cross-attention 적용 후)
            if hasattr(self.decoder, 'output_projection'):
                decoder_outputs['logits'] = self.decoder.output_projection(enhanced_decoder)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Reconstruction loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD)
            recon_loss = loss_fct(
                decoder_outputs['logits'].reshape(-1, decoder_outputs['logits'].size(-1)),
                labels.reshape(-1)
            )
            
            # Relation loss (if cross-attention used)
            relation_loss = 0
            if relation_logits is not None:
                # 자기 관계는 identity (class 0)여야 함
                batch_identity = torch.zeros(batch_size, dtype=torch.long, device=device)
                relation_loss = F.cross_entropy(relation_logits, batch_identity) * 0.1
            
            loss = recon_loss + relation_loss
        
        return {
            'loss': loss,
            'logits': decoder_outputs['logits'],
            'decoder_logits': decoder_outputs['logits'],  # Add for compatibility
            'encoder_hidden_states': encoder_hidden,
            'decoder_hidden_states': decoder_hidden,
            'pooled_output': encoder_outputs['pooled_output'],
            'cross_attention': cross_attn_outputs['cross_attention'] if cross_attn_outputs else None,
            'relation_logits': relation_logits,
            'all_encoder_states': encoder_outputs.get('all_hidden_states', None),
            # Add boundary predictions for visualization
            'char_boundaries': encoder_outputs.get('char_boundaries'),
            'eojeol_boundaries': encoder_outputs.get('eojeol_boundaries'),
            'phrase_boundaries': encoder_outputs.get('phrase_boundaries'),
            'discovered_patterns': encoder_outputs.get('discovered_patterns')
        }
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode single text to representation"""
        tokenized = self.tokenizer.encode(text)
        # Move to same device as model
        device = next(self.parameters()).device
        input_ids = tokenized['input_ids'].unsqueeze(0).to(device)
        attention_mask = tokenized['attention_mask'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask)
        
        return outputs['pooled_output'].squeeze(0)
    
    def decode_representation(self, representation: torch.Tensor, max_length: int = 128) -> str:
        """Decode representation back to text"""
        if representation.dim() == 1:
            representation = representation.unsqueeze(0).unsqueeze(0)
        elif representation.dim() == 2:
            representation = representation.unsqueeze(1)
        
        with torch.no_grad():
            output_ids = self.decoder.generate(representation, max_length=max_length)
        
        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return text
    
    def compute_relation(self, text1: str, text2: str) -> torch.Tensor:
        """Compute relation between two texts"""
        # Encode both texts
        enc1 = self.encode_text(text1).unsqueeze(0).unsqueeze(0)
        enc2 = self.encode_text(text2).unsqueeze(0).unsqueeze(0)
        
        # Compute cross-attention and relations
        with torch.no_grad():
            outputs = self.cross_attention(enc1, enc2)
        
        return F.softmax(outputs['relation_logits'], dim=-1)