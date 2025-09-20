"""
Boundary-Aware Intelligent Tokenizer Model
바이트-문자 관계를 명시적으로 학습하는 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

# Import necessary components from unified_model
from .unified_model import ByteEncoder, TransformerDecoder, CrossAttention, PositionalEncoding


class BoundaryAwareEncoder(nn.Module):
    """
    바이트-문자 경계를 명시적으로 학습하는 인코더
    """
    
    def __init__(
        self,
        vocab_size: int = 260,
        hidden_dims: List[int] = [512, 512, 640, 768, 768],  # 384→512로 증가
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        # 1. 바이트 임베딩
        self.byte_embedding = nn.Embedding(vocab_size, hidden_dims[0])
        
        # 2. 경계 임베딩 (START, CONT, END, SPECIAL) - 더 큰 차원
        self.boundary_embedding = nn.Embedding(4, 128)  # 고정 128차원
        
        # 3. 문자 타입 임베딩 (ASCII, Korean, Chinese, etc.) - 더 큰 차원  
        self.char_type_embedding = nn.Embedding(14, 128)  # 고정 128차원
        
        # 4. 바이트 카운트 임베딩 (1-4 bytes) - UTF-8 패턴 중요
        self.byte_count_embedding = nn.Embedding(5, 128)  # 고정 128차원
        
        # 5. 문자 인덱스 임베딩 (relative position within char)
        self.char_position_embedding = nn.Embedding(4, 128)  # 고정 128차원
        
        # 통합 projection (바이트 임베딩 512 + 구조 임베딩 512 = 1024)
        structural_dim = 128 * 4  # boundary(128) + char_type(128) + byte_count(128) + char_pos(128)
        self.input_projection = nn.Linear(hidden_dims[0] + structural_dim, hidden_dims[0])
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dims[0], max_seq_len, dropout)
        
        # Transformer layers (기존 구조 재사용)
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            input_dim = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            output_dim = hidden_dims[i]
            
            if input_dim != output_dim:
                proj = nn.Linear(input_dim, output_dim)
            else:
                proj = None
            
            layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=num_heads,
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
        
        # Hierarchical Merging Components (새로 추가)
        # 각 레이어마다 병합 모듈 추가 - 트랜스포머가 스스로 결정
        self.merging_modules = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            dim = hidden_dims[i]
            # Learned merging decision - no fixed ratios!
            merge_module = nn.ModuleDict({
                # 경계 학습을 위한 모듈
                'boundary_detector': nn.Linear(dim, 3),  # START, CONT, END
                'merge_attention': nn.MultiheadAttention(dim, num_heads//2, dropout, batch_first=True),
                'merge_gate': nn.Sequential(
                    nn.Linear(dim * 2, dim),
                    nn.ReLU(),
                    nn.Linear(dim, 1)
                ),  # 병합 결정 (학습으로 결정)
                'merge_proj': nn.Linear(dim * 2, dim),  # 병합 후 프로젝션
            })
            self.merging_modules.append(merge_module)
        
        # 경계 예측 헤드
        self.boundary_predictor = nn.Linear(hidden_dims[-1], 4)
        
        # 문자 타입 예측 헤드
        self.char_type_predictor = nn.Linear(hidden_dims[-1], 14)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
        char_types: Optional[torch.Tensor] = None,
        byte_counts: Optional[torch.Tensor] = None,
        char_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. 바이트 임베딩
        byte_emb = self.byte_embedding(input_ids)  # [B, S, D]
        
        # 2. 경계 정보 임베딩 (학습 시에만)
        if boundary_labels is not None:
            boundary_emb = self.boundary_embedding(boundary_labels)  # [B, S, D/4]
        else:
            # 추론 시: 바이트 값으로부터 경계 추정
            # UTF-8 패턴: 
            # 0xxxxxxx (0-127): ASCII (START)
            # 110xxxxx (192-223): 2-byte start
            # 1110xxxx (224-239): 3-byte start
            # 11110xxx (240-247): 4-byte start
            # 10xxxxxx (128-191): continuation
            
            estimated_boundaries = torch.zeros_like(input_ids)
            
            # ASCII (0-127)
            ascii_mask = input_ids < 128
            estimated_boundaries[ascii_mask] = 1  # START
            
            # Continuation bytes (128-191)
            cont_mask = (input_ids >= 128) & (input_ids < 192)
            estimated_boundaries[cont_mask] = 0  # CONT
            
            # Multi-byte starters
            mb_start_mask = input_ids >= 192
            estimated_boundaries[mb_start_mask] = 1  # START
            
            boundary_emb = self.boundary_embedding(estimated_boundaries)
        
        # 3. 문자 타입 임베딩
        if char_types is not None:
            char_type_emb = self.char_type_embedding(char_types)
        else:
            # 추론 시: 기본값 사용
            char_type_emb = self.char_type_embedding(torch.zeros_like(input_ids))
        
        # 4. 바이트 카운트 임베딩
        if byte_counts is not None:
            byte_count_emb = self.byte_count_embedding(torch.clamp(byte_counts, 0, 4))
        else:
            # 추론 시: 바이트 패턴으로 추정
            estimated_counts = torch.ones_like(input_ids)
            # UTF-8 패턴으로 멀티바이트 길이 추정
            estimated_counts[input_ids >= 240] = 4  # 4-byte
            estimated_counts[(input_ids >= 224) & (input_ids < 240)] = 3  # 3-byte
            estimated_counts[(input_ids >= 192) & (input_ids < 224)] = 2  # 2-byte
            byte_count_emb = self.byte_count_embedding(estimated_counts)
        
        # 5. 문자 내 위치 임베딩
        if char_indices is not None:
            # 같은 문자 내에서의 상대 위치 계산
            char_positions = torch.zeros_like(char_indices)
            for b in range(batch_size):
                current_char = -1
                position = 0
                for i in range(seq_len):
                    if char_indices[b, i] != current_char:
                        current_char = char_indices[b, i]
                        position = 0
                    else:
                        position += 1
                    char_positions[b, i] = min(position, 3)
            
            char_pos_emb = self.char_position_embedding(char_positions)
        else:
            char_pos_emb = self.char_position_embedding(torch.zeros_like(input_ids))
        
        # 6. 모든 임베딩 통합
        # 바이트 임베딩 + 구조 정보
        structural_emb = torch.cat([
            boundary_emb,
            char_type_emb,
            byte_count_emb,
            char_pos_emb
        ], dim=-1)  # [B, S, D]
        
        combined_emb = torch.cat([byte_emb, structural_emb], dim=-1)  # [B, S, 2*D]
        
        # Projection to original dimension
        x = self.input_projection(combined_emb)  # [B, S, D]
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Transformer layers with hierarchical merging
        all_hidden_states = []
        boundary_predictions = []
        char_type_predictions = []
        merge_info = []  # 병합 정보 저장
        
        for i, layer_dict in enumerate(self.layers):
            # Project if needed
            if layer_dict['projection'] is not None:
                x = layer_dict['projection'](x)
            
            # Transformer layer
            if attention_mask is not None:
                # Ensure mask matches current sequence length
                current_seq_len = x.size(1)
                if attention_mask.size(1) != current_seq_len:
                    # Adjust mask to match current sequence length after merging
                    key_padding_mask = torch.zeros(x.size(0), current_seq_len, dtype=torch.bool, device=x.device)
                    # Copy valid mask values
                    valid_len = min(attention_mask.size(1), current_seq_len)
                    key_padding_mask[:, :valid_len] = (attention_mask[:, :valid_len] == 0)
                else:
                    key_padding_mask = (attention_mask == 0)
                x = layer_dict['transformer'](x, src_key_padding_mask=key_padding_mask)
            else:
                x = layer_dict['transformer'](x)
            
            x = layer_dict['norm'](x)
            
            # Store hidden state BEFORE merging (for proper gradient flow)
            all_hidden_states.append(x.clone())
            
            # Hierarchical Progressive Merging - 계층적 점진적 병합
            # Layer별로 다른 수준의 병합 학습 (바이트→문자→단어→어절)
            if i < len(self.merging_modules) and self.merging_modules[i] is not None:
                merge_module = self.merging_modules[i]
                batch_size, seq_len, hidden_dim = x.shape
                
                # Skip if already compressed too much
                if seq_len < 4:
                    continue
                
                # Layer 0: UTF-8 경계 기반 병합 (바이트 → 문자)
                if i == 0 and input_ids is not None:
                    # UTF-8 경계 감지를 사용한 확실한 병합
                    merge_decisions = torch.zeros(batch_size, seq_len - 1, device=x.device)
                    
                    for b in range(batch_size):
                        for idx in range(seq_len - 1):
                            if idx < input_ids.shape[1] - 1:
                                current_byte = input_ids[b, idx].item()
                                next_byte = input_ids[b, idx + 1].item()
                                
                                # Continuation byte (10xxxxxx) should merge with previous
                                if 128 <= next_byte < 192:  # Next is continuation
                                    merge_decisions[b, idx] = 1.0  # Merge with next
                                # Special tokens don't merge
                                elif current_byte >= 256 or next_byte >= 256:
                                    merge_decisions[b, idx] = 0.0
                    
                    # Also calculate merge_probs for logging
                    x_pairs = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)
                    merge_scores = merge_module['merge_gate'](x_pairs).squeeze(-1)
                    merge_probs = torch.sigmoid(merge_scores)
                    
                    # Use UTF-8 based decisions for layer 0
                    layer_merge_threshold = 0.5  # Not used but logged
                    
                else:
                    # Other layers: 학습 기반 병합
                    # 1. 트랜스포머가 병합 경계를 학습
                    # 인접 토큰 쌍의 병합 점수 계산
                    x_pairs = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)  # [B, S-1, 2*D]
                    merge_scores = merge_module['merge_gate'](x_pairs).squeeze(-1)  # [B, S-1]
                    merge_probs = torch.sigmoid(merge_scores)  # 0~1 확률
                    
                    # 3. 계층별 병합 강도 설정 (학습 가능)
                    # 중간 레이어: 중간 병합률 (문자→단어)  
                    # 최종 레이어: 높은 병합률 (단어→어절)
                    layer_merge_threshold = 0.7 + (i / len(self.merging_modules)) * 0.2  # 0.7 → 0.9
                    
                    # 4. 병합 결정 (학습된 확률 기반)
                    merge_decisions = (merge_probs > layer_merge_threshold).float()
                
                # 2. Self-attention으로 전역 컨텍스트 파악
                attn_output, attn_weights = merge_module['merge_attention'](x, x, x)
                
                # 5. 실제 병합 수행 (GPU 병렬 처리)
                # 병합 마스크 생성
                merged_indices = []
                merged_x = []
                new_mask = []
                
                # Efficient parallel merging using cumsum trick
                # merge_decisions가 1인 위치에서 다음 토큰과 병합
                # group_ids는 seq_len 크기여야 함 (merge_decisions는 seq_len-1)
                group_ids = torch.zeros(batch_size, seq_len, device=x.device)
                group_ids[:, 0] = 0
                group_ids[:, 1:] = 1 - merge_decisions  # 새 그룹 시작 위치
                group_ids = group_ids.cumsum(dim=1).long()  # 그룹 ID 할당
                
                # 각 그룹의 최대 ID 찾기
                max_groups = group_ids.max(dim=1)[0] + 1  # 각 배치의 그룹 수
                max_group_size = max_groups.max().item()
                
                # 그룹별 aggregation (gradient-safe 방법)
                # Use index_add instead of scatter for better gradient flow
                new_x_list = []
                new_mask_list = []
                
                for b in range(batch_size):
                    # Create mapping from old to new indices
                    unique_groups, inverse_indices = torch.unique(group_ids[b], return_inverse=True)
                    num_groups = len(unique_groups)
                    
                    # Initialize new tensor for this batch
                    batch_new_x = torch.zeros(num_groups, hidden_dim, device=x.device)
                    group_counts = torch.zeros(num_groups, device=x.device)
                    
                    # Sum tokens belonging to same group
                    batch_new_x = batch_new_x.index_add(0, inverse_indices, x[b])
                    group_counts = group_counts.index_add(0, inverse_indices, torch.ones(seq_len, device=x.device))
                    
                    # Average
                    batch_new_x = batch_new_x / group_counts.unsqueeze(-1).clamp(min=1)
                    
                    new_x_list.append(batch_new_x)
                    new_mask_list.append(torch.ones(num_groups, device=x.device))
                
                # Pad to same size for batching
                max_new_len = max(t.size(0) for t in new_x_list)
                padded_x_list = []
                padded_mask_list = []
                
                for batch_x, batch_mask in zip(new_x_list, new_mask_list):
                    pad_len = max_new_len - batch_x.size(0)
                    if pad_len > 0:
                        batch_x = torch.cat([batch_x, torch.zeros(pad_len, hidden_dim, device=x.device)], dim=0)
                        batch_mask = torch.cat([batch_mask, torch.zeros(pad_len, device=x.device)], dim=0)
                    padded_x_list.append(batch_x)
                    padded_mask_list.append(batch_mask)
                
                new_x = torch.stack(padded_x_list)
                valid_mask = torch.stack(padded_mask_list)
                
                # Trim to actual size (important for gradient flow)
                actual_len = valid_mask.sum(dim=1).max().long().item()
                new_x = new_x[:, :actual_len]
                valid_mask = valid_mask[:, :actual_len]
                
                # Attention 정보 추가 (선택적)
                new_x = new_x + attn_output.mean(dim=1, keepdim=True).expand(-1, actual_len, -1) * 0.1
                
                # Update x and attention_mask
                x = new_x
                attention_mask = valid_mask
                
                # Note: DO NOT re-apply positional encoding after merging
                # The transformer already learned position-aware representations
                
                # Store merge mapping for cross-attention and decoder
                # 원본 위치 → 병합 후 위치 매핑 저장 (디코더 복원용)
                merge_mapping = {
                    'original_positions': torch.arange(seq_len, device=x.device),
                    'merged_groups': group_ids,
                    'group_sizes': None  # No longer using counts
                }
                
                # 정보 기록 (actual_len already computed above)
                merge_info.append({
                    'layer': i,
                    'original_len': seq_len,
                    'merged_len': actual_len,
                    'compression_ratio': seq_len / max(actual_len, 1),
                    'merge_threshold': layer_merge_threshold,
                    'avg_merge_prob': merge_probs.mean().item(),
                    'merge_mapping': merge_mapping  # 복원을 위한 매핑 정보
                })
            
            # 중간 층에서도 경계 예측 (auxiliary loss) - 마지막 층에서만
            if i == len(self.layers) - 1:  # 마지막 층에서만 예측
                boundary_pred = self.boundary_predictor(x)
                char_type_pred = self.char_type_predictor(x)
                boundary_predictions.append(boundary_pred)
                char_type_predictions.append(char_type_pred)
        
        # Pool for sequence representation
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = x.mean(dim=1)
        
        return {
            'last_hidden_state': x,
            'pooled_output': pooled,
            'all_hidden_states': all_hidden_states,
            'boundary_predictions': boundary_predictions,  # 경계 예측 (여러 층)
            'char_type_predictions': char_type_predictions,  # 문자 타입 예측
            'boundary_logits': self.boundary_predictor(x),  # 최종 경계 예측
            'char_type_logits': self.char_type_predictor(x),  # 최종 문자 타입 예측
            'merge_info': merge_info,  # 병합 정보 (새로 추가)
            'attention_mask': attention_mask  # 업데이트된 마스크 반환
        }


class BoundaryAwareTokenizerModel(nn.Module):
    """
    바이트-문자 관계를 명시적으로 학습하는 통합 모델
    """
    
    def __init__(
        self,
        vocab_size: int = 260,
        encoder_dims: List[int] = [512, 512, 640, 768, 768],  # 384→512로 증가
        decoder_hidden: int = 768,
        num_heads: int = 8,
        num_decoder_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        
        # Boundary-aware encoder
        self.encoder = BoundaryAwareEncoder(
            vocab_size, encoder_dims, num_heads, dropout, max_seq_len
        )
        
        # Standard decoder (재사용)
        self.decoder = TransformerDecoder(
            vocab_size, decoder_hidden, num_heads, num_decoder_layers, dropout, max_seq_len
        )
        
        # Cross-attention (재사용)
        self.cross_attention = CrossAttention(encoder_dims[-1], num_heads, dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        boundary_labels: Optional[torch.Tensor] = None,
        char_types: Optional[torch.Tensor] = None,
        byte_counts: Optional[torch.Tensor] = None,
        char_indices: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cross_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Boundary-aware encoding
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            boundary_labels=boundary_labels,
            char_types=char_types,
            byte_counts=byte_counts,
            char_indices=char_indices,
            attention_mask=attention_mask
        )
        
        encoder_hidden = encoder_outputs['last_hidden_state']
        
        # 2. Decoding
        # Pass the updated attention_mask from encoder (after merging)
        encoder_mask = encoder_outputs.get('attention_mask', attention_mask)
        
        # Use input_ids as decoder_input_ids for teacher forcing if not provided
        if decoder_input_ids is None and input_ids is not None:
            decoder_input_ids = input_ids
        
        decoder_outputs = self.decoder(
            encoder_hidden,
            decoder_input_ids,
            encoder_mask  # Use encoder's updated mask
        )
        
        # 3. Cross-attention (optional)
        cross_attn_outputs = None
        relation_logits = None
        
        if use_cross_attention and decoder_outputs['hidden_states'] is not None:
            decoder_hidden = decoder_outputs['hidden_states']
            
            cross_attn_outputs = self.cross_attention(
                query=decoder_hidden,
                key=encoder_hidden,
                query_mask=None,
                key_mask=attention_mask
            )
            
            relation_logits = cross_attn_outputs['relation_logits']
            
            # Enhanced decoder with cross-attention
            enhanced_decoder = decoder_hidden + cross_attn_outputs['cross_attention']
            decoder_outputs['logits'] = self.decoder.output_projection(enhanced_decoder)
        
        # 4. Loss calculation
        total_loss = None
        if labels is not None:
            # Reconstruction loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=256)  # PAD
            recon_loss = loss_fct(
                decoder_outputs['logits'].reshape(-1, decoder_outputs['logits'].size(-1)),
                labels.reshape(-1)
            )
            
            total_loss = recon_loss
            
            # Boundary prediction loss
            if boundary_labels is not None and 'boundary_logits' in encoder_outputs:
                boundary_logits = encoder_outputs['boundary_logits']
                # Check if dimensions match
                logits_size = boundary_logits.size(0) * boundary_logits.size(1)
                labels_size = boundary_labels.numel()
                
                if logits_size == labels_size:
                    boundary_loss_fct = nn.CrossEntropyLoss(ignore_index=3)  # special
                    boundary_loss = boundary_loss_fct(
                        boundary_logits.reshape(-1, 4),
                        boundary_labels.reshape(-1)
                    )
                    total_loss = total_loss + boundary_loss * 0.3
                # If encoder changed sequence length (due to merging), skip boundary loss
                # This is expected behavior when boundary-aware merging is active
            
            # Character type prediction loss
            if char_types is not None and 'char_type_logits' in encoder_outputs:
                char_type_logits = encoder_outputs['char_type_logits']
                # Check if dimensions match
                logits_size = char_type_logits.size(0) * char_type_logits.size(1)
                labels_size = char_types.numel()
                
                if logits_size == labels_size:
                    char_type_loss_fct = nn.CrossEntropyLoss(ignore_index=13)  # special
                    char_type_loss = char_type_loss_fct(
                        char_type_logits.reshape(-1, 14),
                        char_types.reshape(-1)
                    )
                    total_loss = total_loss + char_type_loss * 0.2
                # If encoder changed sequence length (due to merging), skip char type loss
            
            # Auxiliary losses from intermediate layers
            if encoder_outputs.get('boundary_predictions') and boundary_labels is not None:
                # boundary_loss_fct는 위에서 정의된 경우에만 사용
                if 'boundary_loss_fct' in locals():
                    for boundary_pred in encoder_outputs['boundary_predictions']:
                        # Ensure batch sizes match
                        pred_batch_size = boundary_pred.size(0) * boundary_pred.size(1)
                        label_batch_size = boundary_labels.numel()
                        
                        if pred_batch_size == label_batch_size:
                            aux_boundary_loss = boundary_loss_fct(
                                boundary_pred.reshape(-1, 4),
                                boundary_labels.reshape(-1)
                            )
                            total_loss = total_loss + aux_boundary_loss * 0.1
                        else:
                            # Skip if dimensions don't match (different layer sizes)
                            continue
        
        return {
            'loss': total_loss,
            'logits': decoder_outputs['logits'],
            'encoder_hidden_states': encoder_hidden,
            'decoder_hidden_states': decoder_outputs['hidden_states'],
            'boundary_logits': encoder_outputs['boundary_logits'],
            'char_type_logits': encoder_outputs['char_type_logits'],
            'boundary_predictions': encoder_outputs.get('boundary_predictions'),
            'relation_logits': relation_logits,
            'cross_attention': cross_attn_outputs['cross_attention'] if cross_attn_outputs else None
        }