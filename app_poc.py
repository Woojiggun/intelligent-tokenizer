#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Intelligent Tokenizer v6.0 - POC Demo
Using black-box inference model (core architecture hidden)
"""

import gradio as gr
import torch
import time
from huggingface_hub import hf_hub_download
from inference_model import IntelligentTokenizerInference


class ByteTokenizerV6:
    """Basic byte tokenizer for preprocessing"""
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
            return ""
            
        try:
            result = bytes(filtered).decode('utf-8', errors='ignore')
            return result
        except:
            return ""


# Global variables
device = torch.device('cpu')
model = None
tokenizer = ByteTokenizerV6()
model_loaded = False


def load_model():
    """Load the black-box inference model"""
    global model, model_loaded
    
    try:
        # Download model from Hugging Face
        model_file = hf_hub_download(
            repo_id="ggunio/intelligent-tokenizer-v6",
            filename="pytorch_model.bin"
        )
        
        # Create inference model
        model = IntelligentTokenizerInference()
        
        # Load weights
        state_dict = torch.load(model_file, map_location=device)
        model.load_weights(state_dict)
        model.eval()
        model = model.to(device)
        
        model_loaded = True
        return "✅ Model loaded successfully"
    except Exception as e:
        return f"❌ Model loading failed: {e}"


# Initialize on startup
load_status = load_model()
print(load_status)


def process_text(text):
    """Process text with the model"""
    if not text:
        return "Please enter text to process."
    
    if not model_loaded:
        return "Model not loaded. Please refresh the page."
    
    start_time = time.time()
    results = []
    
    # Encode text
    chunks = tokenizer.encode(text)
    num_bytes = len(text.encode('utf-8'))
    num_chunks = len(chunks)
    
    # Process each chunk
    for i, chunk in enumerate(chunks[:3]):  # Limit to 3 chunks for demo
        try:
            input_ids = torch.tensor([chunk], device=device)
            
            with torch.no_grad():
                # Teacher forcing for reconstruction test
                decoder_input = input_ids[:, :-1]
                labels = input_ids[:, 1:]
                
                # Run inference
                outputs = model(input_ids, decoder_input)
                logits = outputs['logits']
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Calculate accuracy
                accuracy = (predictions == labels).float().mean().item()
                
                # Decode results
                original_text = tokenizer.decode(chunk[1:-1])
                recovered_text = tokenizer.decode(predictions[0].cpu().tolist())
                
                results.append({
                    'chunk': i + 1,
                    'original': original_text[:100],
                    'recovered': recovered_text[:100],
                    'accuracy': accuracy
                })
        except Exception as e:
            results.append({
                'chunk': i + 1,
                'original': tokenizer.decode(chunk[1:-1])[:100],
                'recovered': f"[Error: {str(e)[:50]}]",
                'accuracy': 0.0
            })
    
    latency = (time.time() - start_time) * 1000
    
    # Language detection
    lang_info = detect_language(text)
    
    # Format output
    output = f"""# 📊 Processing Results

## Input Text
```
{text[:500] + '...' if len(text) > 500 else text}
```

## 📈 Statistics
- **Characters**: {len(text)}
- **Bytes**: {num_bytes}
- **Chunks**: {num_chunks} (256-byte limit)
- **Processing Time**: {latency:.1f}ms

## 🔍 Language Detection
{lang_info}

## 🔄 Reconstruction Test
"""
    
    for r in results:
        status = "✅" if r['accuracy'] > 0.9 else "⚠️" if r['accuracy'] > 0.5 else "❌"
        output += f"""
### Chunk {r['chunk']}
- **Original**: {r['original']}
- **Recovered**: {r['recovered']}
- **Accuracy**: {r['accuracy']:.1%} {status}
"""
    
    output += """
## 📊 Expected Performance (Based on Training)
- English/European: 95-100% ✅
- Korean: 70% ⚠️
- Japanese: 81% ✅
- Chinese: 7% ❌
- Rare Languages: 47% average

## ℹ️ Model Information
- **Architecture**: Proprietary (black-box inference)
- **Parameters**: 105M
- **Training**: 22 epochs, Flores-200 dataset
- **Status**: POC demonstration

---
*Note: Core architecture is proprietary. May open source based on community interest.*
*Contact: ggunio5782@gmail.com*
"""
    return output


def detect_language(text):
    """Simple language detection"""
    patterns = []
    
    if any(ord('가') <= ord(c) <= ord('힣') for c in text):
        patterns.append("🇰🇷 **Korean** - Expected: 70%")
    
    if any(c.isalpha() and ord(c) < 128 for c in text):
        patterns.append("🇬🇧 **English** - Expected: 95-100%")
    
    if any(0x4E00 <= ord(c) <= 0x9FFF for c in text):
        patterns.append("🇨🇳 **Chinese** - Expected: 7%")
    
    if any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF for c in text):
        patterns.append("🇯🇵 **Japanese** - Expected: 81%")
    
    return '\n'.join(patterns) if patterns else "Language not detected"


# Gradio interface
demo = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        label="Input Text",
        placeholder="Enter text to process...",
        lines=3
    ),
    outputs=gr.Markdown(label="Analysis Results"),
    examples=[
        "The quick brown fox jumps over the lazy dog",
        "안녕하세요. 오늘 날씨가 좋네요.",
        "こんにちは。今日はいい天気ですね",
    ],
    title="🚀 Intelligent Tokenizer v6.0 - POC Demo",
    description="""
    **Language Pattern Learning through Pure Neural Networks**
    
    This is a POC demonstration of the Intelligent Tokenizer that learns language patterns from raw bytes.
    
    ## Key Features:
    - No vocabulary files (260 bytes only)
    - Learns patterns from data
    - Works with 204 languages
    - Designed for LLM communication
    
    ## Important Notes:
    - Core architecture is proprietary (black-box model)
    - May open source based on community interest
    - Not production ready (POC stage)
    
    Developer: Woo Jinhyun | Implementation: Claude Code
    """,
    theme="default",
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()