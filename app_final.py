#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent Tokenizer v6.0 - Embedding Communication Layer for LLMs
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
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
        
        # Split into chunks (256 byte limit for real-time processing)
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
        filtered = []
        for id in ids:
            if isinstance(id, (int, torch.Tensor)):
                if torch.is_tensor(id):
                    id = id.item()
                if 0 <= id < 256:
                    filtered.append(id)
        
        if not filtered:
            return "[Empty result]"
            
        try:
            result = bytes(filtered).decode('utf-8', errors='ignore')
            return result if result else "[Decoding failed]"
        except Exception as e:
            return f"[Error: {str(e)}]"

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
    
    def get_embeddings(self, input_ids):
        """Extract embeddings for LLM communication"""
        with torch.no_grad():
            encoder_output = self.forward(input_ids)
            # Average pooling for fixed-size embedding
            embeddings = encoder_output.mean(dim=1)  # [batch, hidden_size]
            return embeddings

# Global variables
device = torch.device('cpu')  # HF Space uses CPU
model = None
tokenizer = ByteTokenizerV6()

def load_model():
    global model
    try:
        # Download model from Hugging Face
        model_file = hf_hub_download(
            repo_id="ggunio/intelligent-tokenizer-v6",
            filename="pytorch_model.bin"
        )
        
        # Initialize model
        model = BoundaryAwareTokenizerModel(
            vocab_size=260,
            hidden_size=768,
            num_encoder_layers=5,
            num_decoder_layers=6,
            num_heads=8,
            dropout=0.1,
            max_position_embeddings=256
        ).to(device)
        
        # Load weights
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        return "✅ Model loaded successfully"
    except Exception as e:
        return f"❌ Model loading failed: {e}"

# Load model on startup
load_status = load_model()
print(load_status)

def process_text(text, show_embeddings=False):
    """Process text and generate embeddings"""
    if not text:
        return "Please enter text to process."
    
    if model is None:
        return "Model not loaded. Please refresh the page."
    
    start_time = time.time()
    
    # 1. Encode text
    chunks = tokenizer.encode(text)
    num_bytes = len(text.encode('utf-8'))
    num_chunks = len(chunks)
    
    # 2. Process with model
    results = []
    embeddings_list = []
    
    for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks for demo
        input_ids = torch.tensor([chunk], device=device)
        
        with torch.no_grad():
            # Get embeddings
            embeddings = model.get_embeddings(input_ids)
            embeddings_list.append(embeddings)
            
            # Try reconstruction
            if len(chunk) > 1:
                decoder_input = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                
                logits = model(input_ids, decoder_input)
                predictions = torch.argmax(logits, dim=-1)
                
                # Decode predictions
                recovered_bytes = predictions[0].cpu().tolist()
                recovered_text = tokenizer.decode(recovered_bytes)
                
                # Calculate accuracy
                accuracy = (predictions == labels).float().mean().item()
                
                results.append({
                    'chunk_id': i + 1,
                    'original': tokenizer.decode(chunk[1:-1]),  # Remove BOS/EOS
                    'recovered': recovered_text,
                    'accuracy': accuracy,
                    'embedding_shape': embeddings.shape
                })
    
    latency = (time.time() - start_time) * 1000
    
    # 3. Language detection
    lang_info = detect_language(text)
    
    # 4. Format results
    result_text = f"""
# 📊 Processing Results

## Input Text
```
{text[:500] + '...' if len(text) > 500 else text}
```

## 📈 Basic Information
- **Characters**: {len(text)}
- **Bytes**: {num_bytes}
- **Chunks**: {num_chunks} (256-byte limit per chunk)
- **Processing Time**: {latency:.1f}ms

## 🔍 Language Detection
{lang_info}

## 🧬 Embedding Generation for LLM Communication
"""
    
    if show_embeddings and embeddings_list:
        combined_embedding = torch.cat(embeddings_list, dim=0).mean(dim=0)
        result_text += f"""
### Embedding Statistics:
- **Dimension**: 768 (compatible with BERT/GPT architectures)
- **Norm**: {combined_embedding.norm().item():.4f}
- **Mean**: {combined_embedding.mean().item():.6f}
- **Std**: {combined_embedding.std().item():.6f}

### Sample Embedding Values (first 10 dimensions):
```python
{combined_embedding[:10].cpu().numpy().round(4).tolist()}
```

### Use Cases:
1. **LLM Communication**: Direct embedding input for language models
2. **RAG Systems**: Semantic search and retrieval
3. **Similarity Computation**: Cross-lingual semantic matching
4. **Clustering**: Language-agnostic document grouping
"""

    result_text += f"""
## 🔄 Reconstruction Test
"""
    
    for r in results:
        result_text += f"""
### Chunk {r['chunk_id']}:
- **Original**: {r['original'][:100]}
- **Recovered**: {r['recovered'][:100]}
- **Accuracy**: {r['accuracy']:.1%}
- **Embedding Shape**: {r['embedding_shape']}
"""

    result_text += f"""
## 🎯 Performance Metrics (Epoch 22)
- English/European: 95-100% ✅
- Korean: 70% 🔄 (jamo merging complexity)
- Japanese: 81% ✅
- Chinese: 7% ⚠️ (needs more training)
- Rare Languages: 47% average

## 💡 Project Vision

### Current Status (POC):
- **Pure learning-based**: No vocabulary files, learns patterns from data
- **Language-agnostic**: Treats all 204 languages equally
- **Embedding-focused**: Designed for LLM communication, not standalone tokenization
- **Hardware limitation**: RTX 4070 (training is slow, compression learning in progress)

### Future Development:
1. **RAG Integration**: Semantic search layer for retrieval-augmented generation
2. **Inference Layer**: Add reasoning capabilities for on-device deployment
3. **Lightweight Models**: Target mobile and edge devices (under 100MB)
4. **Open Source**: Planning to release code based on community interest

### Key Innovation:
Unlike traditional tokenizers that focus on vocabulary compression, this system:
- **Learns semantic units** from raw bytes
- **Preserves meaning** across languages
- **Generates embeddings** directly for LLM communication
- **Enables cross-lingual** understanding without translation

---
*Note: This is a research POC. Commercial performance not guaranteed.*
*Contact: ggunio5782@gmail.com | LinkedIn: www.linkedin.com/in/namuneup*
"""
    return result_text

def detect_language(text):
    """Simple language detection"""
    patterns = []
    
    # Korean
    if any(ord('가') <= ord(c) <= ord('힣') for c in text):
        patterns.append("🇰🇷 **Korean** - Expected accuracy: 70%")
    
    # English
    if any(c.isalpha() and ord(c) < 128 for c in text):
        patterns.append("🇬🇧 **English** - Expected accuracy: 95-100%")
    
    # Chinese
    if any(0x4E00 <= ord(c) <= 0x9FFF for c in text):
        patterns.append("🇨🇳 **Chinese** - Expected accuracy: 7%")
    
    # Japanese
    if any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF for c in text):
        patterns.append("🇯🇵 **Japanese** - Expected accuracy: 81%")
    
    # Arabic
    if any(0x0600 <= ord(c) <= 0x06FF for c in text):
        patterns.append("🇸🇦 **Arabic** - Expected accuracy: 38%")
    
    # Russian
    if any(0x0400 <= ord(c) <= 0x04FF for c in text):
        patterns.append("🇷🇺 **Russian** - Expected accuracy: 63%")
    
    return '\n'.join(patterns) if patterns else "Language not detected"

# Example texts
examples = [
    ["The quick brown fox jumps over the lazy dog", True],
    ["안녕하세요. 오늘 날씨가 좋네요.", True],
    ["こんにちは。今日はいい天気ですね", False],
    ["Bonjour. Comment allez-vous?", False],
    ["今天天气很好，我们去公园散步吧", False],
]

# Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=[
        gr.Textbox(
            label="Input Text",
            placeholder="Enter text to process...",
            lines=3
        ),
        gr.Checkbox(
            label="Show Embedding Analysis",
            value=False,
            info="Display embedding vectors for LLM communication"
        )
    ],
    outputs=gr.Markdown(label="Analysis Results"),
    examples=examples,
    title="🚀 Intelligent Tokenizer v6.0 - Embedding Communication Layer",
    description="""
    **Language Pattern Learning through Pure Neural Networks**
    
    This is NOT just another tokenizer - it's an embedding communication layer designed for LLM integration.
    
    ## 🎯 Core Purpose:
    - **LLM Communication**: Generate semantic embeddings for language models
    - **RAG Systems**: Enable semantic search and retrieval
    - **On-device AI**: Future lightweight models for edge deployment
    
    ## 📊 Technical Details:
    - 4 months development (Aug-Dec 2024)
    - Designer: Woo Jinhyun | Implementation: Claude Code collaboration
    - Training: RTX 4070, 22 epochs on Flores-200 (204 languages)
    - Model: 105M parameters (5-layer encoder + 6-layer decoder)
    - No vocabulary files - pure byte-level learning (260 tokens only)
    
    ## 🚧 Current Limitations:
    - POC stage - not production-ready
    - 256-byte chunks for real-time processing
    - Compression learning still in progress (hardware constraints)
    - Variable accuracy across languages
    
    ## 🔮 Future Plans:
    - Add inference layer for reasoning
    - Optimize for mobile deployment (<100MB)
    - **Open source release planned** (based on community interest)
    - Integration with popular LLM frameworks
    
    GitHub: [Coming Soon - depends on interest] | Paper: [In Progress]
    """,
    theme="default",
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()