#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent Tokenizer - Hugging Face Space Demo
"""

import gradio as gr
import torch
import torch.nn as nn
import time
from huggingface_hub import hf_hub_download

# ByteTokenizer 클래스 정의
class ByteTokenizerV6:
    def __init__(self, max_seq_len=256):
        self.vocab_size = 260
        self.max_seq_len = max_seq_len
        self.PAD = 256
        self.BOS = 257
        self.EOS = 258  
        self.MASK = 259
    
    def encode(self, text):
        """텍스트를 바이트 시퀀스로 변환"""
        byte_seq = list(text.encode('utf-8'))
        
        # 청크 단위로 분할 (256 바이트 제한)
        if len(byte_seq) > self.max_seq_len - 2:
            chunks = []
            for i in range(0, len(byte_seq), self.max_seq_len - 2):
                chunk = byte_seq[i:i + self.max_seq_len - 2]
                chunks.append([self.BOS] + chunk + [self.EOS])
            return chunks
        else:
            return [[self.BOS] + byte_seq + [self.EOS]]
    
    def decode(self, ids):
        """바이트 시퀀스를 텍스트로 변환"""
        filtered = [id for id in ids if id < 256]
        try:
            return bytes(filtered).decode('utf-8', errors='replace')
        except:
            return "[디코딩 오류]"

# 모델 아키텍처 정의
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

# 모델 로드
device = torch.device('cpu')  # Hugging Face Space는 기본 CPU
model = None
tokenizer = ByteTokenizerV6()

def load_model():
    global model
    try:
        # Hugging Face에서 모델 다운로드
        model_file = hf_hub_download(
            repo_id="ggunio/intelligent-tokenizer-v6",
            filename="pytorch_model.bin"
        )
        
        # 모델 초기화
        model = BoundaryAwareTokenizerModel(
            vocab_size=260,
            hidden_size=768,
            num_encoder_layers=5,
            num_decoder_layers=6,
            num_heads=8,
            dropout=0.1,
            max_position_embeddings=256
        ).to(device)
        
        # 가중치 로드
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return "✅ 모델 로드 완료"
    except Exception as e:
        return f"❌ 모델 로드 실패: {e}"

# 초기 모델 로드
load_status = load_model()
print(load_status)

def process_text(text):
    """텍스트 처리 및 분석"""
    if not text:
        return "텍스트를 입력하세요."
    
    if model is None:
        return "모델이 로드되지 않았습니다. 잠시 후 다시 시도해주세요."
    
    start_time = time.time()
    
    # 1. 바이트 인코딩 (청크 처리)
    chunks = tokenizer.encode(text)
    num_bytes = len(text.encode('utf-8'))
    num_chunks = len(chunks)
    
    # 2. 모델 처리 (첫 번째 청크만)
    chunk = chunks[0]
    input_ids = torch.tensor([chunk], device=device)
    
    with torch.no_grad():
        # 인코더 처리
        encoded = model(input_ids)
        
        # 디코더로 복원
        if len(chunk) > 1:
            decoder_input = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            
            logits = model(input_ids, decoder_input)
            predictions = torch.argmax(logits, dim=-1)
            
            # 복원 텍스트
            recovered_bytes = predictions[0].tolist()
            recovered_text = tokenizer.decode(recovered_bytes)
            
            # 정확도 계산
            accuracy = (predictions == labels).float().mean().item()
        else:
            recovered_text = text
            accuracy = 1.0
    
    latency = (time.time() - start_time) * 1000
    
    # 언어 감지 및 예상 복원률
    lang_info = detect_language(text)
    
    # 결과 포맷팅
    result = f"""
# 📊 처리 결과

## 입력 텍스트
```
{text[:256] + '...' if len(text) > 256 else text}
```

## 📈 기본 정보
- **문자 수**: {len(text)} 글자
- **바이트 수**: {num_bytes} bytes
- **청크 수**: {num_chunks} {'(256 바이트 제한으로 분할)' if num_chunks > 1 else ''}
- **처리 시간**: {latency:.1f}ms

## 🔍 언어 감지
{lang_info}

## 🔄 복원 결과
```
{recovered_text[:256] + '...' if len(recovered_text) > 256 else recovered_text}
```
- **복원 정확도**: {accuracy:.1%}

## 🎯 성능 지표 (Epoch 22 기준)
- 영어/독일어/프랑스어: 95-100%
- 한국어: 70% (자모 병합 복잡성)
- 일본어: 81%
- 중국어: 7% (추가 학습 필요)
- 희소 언어: 평균 47%

## 💡 핵심 혁신
- ✅ Vocabulary 없음 (260 bytes only)
- ✅ 언어 패턴 자동 학습
- ✅ 의미 단위 보존
- ✅ 모든 언어 평등 처리

---
*Note: 256 바이트 이상의 텍스트는 청크로 분할 처리됩니다.*
"""
    return result

def detect_language(text):
    """간단한 언어 감지"""
    patterns = []
    
    # 한국어
    if any(ord('가') <= ord(c) <= ord('힣') for c in text):
        patterns.append("🇰🇷 **한국어** - 예상 복원률: 70%")
    
    # 영어
    if any(c.isalpha() and ord(c) < 128 for c in text):
        patterns.append("🇬🇧 **영어** - 예상 복원률: 95-100%")
    
    # 중국어
    if any(0x4E00 <= ord(c) <= 0x9FFF for c in text):
        patterns.append("🇨🇳 **중국어** - 예상 복원률: 7%")
    
    # 일본어
    if any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF for c in text):
        patterns.append("🇯🇵 **일본어** - 예상 복원률: 81%")
    
    # 아랍어
    if any(0x0600 <= ord(c) <= 0x06FF for c in text):
        patterns.append("🇸🇦 **아랍어** - 예상 복원률: 38%")
    
    # 러시아어
    if any(0x0400 <= ord(c) <= 0x04FF for c in text):
        patterns.append("🇷🇺 **러시아어** - 예상 복원률: 63%")
    
    return '\n'.join(patterns) if patterns else "언어를 감지할 수 없습니다."

# 예제 텍스트
examples = [
    "The quick brown fox jumps over the lazy dog",  # 영어 100%
    "안녕하세요. 오늘 날씨가 좋네요.",  # 한국어 70%
    "こんにちは。今日はいい天気ですね",  # 일본어 81%
    "Bonjour. Comment allez-vous?",  # 프랑스어 100%
    "Habari! Unaendeleaje leo?",  # 스와힐리어 100%
    "Привет! Как твои дела?",  # 러시아어 63%
    "今天天气很好",  # 중국어 7%
]

# Gradio 인터페이스
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        label="텍스트 입력",
        placeholder="테스트할 텍스트를 입력하세요...",
        lines=3
    ),
    outputs=gr.Markdown(label="분석 결과"),
    examples=examples,
    title="🚀 Intelligent Tokenizer v6.0 - Language Pattern Learning",
    description="""
    **언어 패턴 학습 토크나이저**
    
    - 4개월 개발 (2024.08 - 2024.12)
    - 설계: Woo Jinhyun / 구현: Claude Code 협업
    - RTX 4070 환경에서 22 Epochs 학습
    - 204개 언어 지원 (Flores-200 데이터셋)
    - Vocabulary 파일 없음 (260 bytes only)
    - 순수 학습 기반 (언어별 규칙 없음)
    
    **실측 복원률** (Epoch 22 기준):
    - 영어/유럽어: 95-100% ✅
    - 한국어: 70% 🔄 (자모 병합 개선 중)
    - 일본어: 81% ✅
    - 중국어: 7% ⚠️ (추가 학습 필요)
    - 희소 언어: 평균 47%
    
    **특징**: 한국어 조사, 영어 형태소, 중국어 문자 패턴을 순수 학습으로 발견합니다.
    
    **주의**: 이 프로젝트는 실험적 POC이며, 상용 수준의 성능을 보장하지 않습니다.
    
    GitHub: [Coming Soon] | Paper: [In Progress]
    """,
    theme="default",
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()