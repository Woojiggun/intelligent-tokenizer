#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
다국어 복원률 테스트 - 실제 체크포인트로 측정
"""

import torch
import sys
import io
from pathlib import Path
import json

# UTF-8 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 상위 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "intelligent-tokenizer_v6.0"))

from core.boundary_aware_model import BoundaryAwareTokenizerModel
from src.core.byte_tokenizer_v6 import ByteTokenizerV6

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# 체크포인트 로드
checkpoint_path = Path("../intelligent-tokenizer_v6.0/checkpoints/unified/latest_checkpoint.pt")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model = BoundaryAwareTokenizerModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

tokenizer = ByteTokenizerV6()

print(f"Model loaded: Epoch {checkpoint['epoch']}, Loss {checkpoint['loss']:.4f}\n")
print("="*70)

def test_language(text, lang_name):
    """언어별 테스트"""
    encoded = tokenizer.encode(text)
    byte_ids = encoded['input_ids']
    
    # 256 바이트 제한 체크
    if len(byte_ids) > 256:
        text = text[:80]  # 줄이기
        encoded = tokenizer.encode(text)
        byte_ids = encoded['input_ids']
    
    input_ids = torch.tensor([byte_ids], device=device)
    attention_mask = torch.tensor([encoded['attention_mask']], device=device)
    
    with torch.no_grad():
        # Teacher Forcing 정확도
        if len(byte_ids) > 1:
            decoder_input = input_ids[:, :-1]
            labels = input_ids[:, 1:]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input,
                labels=labels,
                use_cross_attention=True
            )
            
            predictions = torch.argmax(outputs['logits'], dim=-1)
            accuracy = (predictions == labels).float().mean().item()
            loss = outputs['loss'].item()
        else:
            accuracy = 1.0
            loss = 0.0
        
        # Autoregressive 생성
        generated = [tokenizer.BOS]
        max_len = min(len(byte_ids) * 2, 256)
        
        encoder_outputs = model.encoder(input_ids, attention_mask)
        encoder_hidden = encoder_outputs['last_hidden_state']
        
        for _ in range(max_len):
            decoder_input = torch.tensor([generated], dtype=torch.long, device=device)
            decoder_output = model.decoder(
                encoder_hidden,
                decoder_input,
                attention_mask
            )
            logits = decoder_output['logits'] if isinstance(decoder_output, dict) else decoder_output
            next_token = torch.argmax(logits[0, -1]).item()
            generated.append(next_token)
            if next_token == tokenizer.EOS:
                break
        
        # 복원 텍스트
        valid_bytes = [b for b in generated[1:] if b < 256 and b != tokenizer.EOS]
        try:
            reconstructed = bytes(valid_bytes).decode('utf-8', errors='ignore')
            # 문자 단위 정확도
            correct_chars = sum(1 for a, b in zip(text, reconstructed) if a == b)
            char_accuracy = correct_chars / max(len(text), 1)
        except:
            reconstructed = "[decode error]"
            char_accuracy = 0.0
    
    return {
        'text': text,
        'lang': lang_name,
        'tf_accuracy': accuracy,
        'char_accuracy': char_accuracy,
        'loss': loss,
        'reconstructed': reconstructed[:50] if len(reconstructed) > 50 else reconstructed
    }

# 다양한 언어 테스트 샘플
test_samples = {
    "🌏 주요 언어": {
        "English": "The quick brown fox jumps over the lazy dog",
        "한국어": "안녕하세요. 오늘 날씨가 정말 좋네요",
        "中文简体": "今天天气很好，我们去公园散步吧",
        "中文繁體": "今天天氣很好，我們去公園散步吧",
        "日本語": "こんにちは。今日はいい天気ですね",
        "Español": "Buenos días. ¿Cómo está usted?",
        "Français": "Bonjour. Comment allez-vous aujourd'hui?",
        "Deutsch": "Guten Tag. Wie geht es Ihnen heute?",
        "Русский": "Привет! Как твои дела сегодня?",
        "العربية": "مرحبا! كيف حالك اليوم؟",
        "हिन्दी": "नमस्ते। आप कैसे हैं?",
        "Português": "Olá! Como você está hoje?",
    },
    
    "🌍 희소 언어": {
        "Swahili": "Habari! Unaendeleaje leo?",
        "isiZulu": "Sawubona! Unjani namuhla?",
        "አማርኛ": "ሰላም! እንዴት ነህ?",
        "ქართული": "გამარჯობა! როგორ ხარ?",
        "Հայերեն": "Բարև! Ինչպես ես այսօր?",
        "עברית": "שלום! מה שלומך היום?",
        "فارسی": "سلام! حال شما چطور است؟",
        "বাংলা": "হ্যালো! আপনি কেমন আছেন?",
        "தமிழ்": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "ไทย": "สวัสดี! คุณเป็นอย่างไรบ้าง?",
        "Tiếng Việt": "Xin chào! Bạn khỏe không?",
        "Bahasa Indonesia": "Halo! Apa kabar?",
        "Türkçe": "Merhaba! Nasılsınız?",
        "Polski": "Cześć! Jak się masz?",
        "Nederlands": "Hallo! Hoe gaat het?",
        "Ελληνικά": "Γεια σου! Πώς είσαι;",
        "Čeština": "Ahoj! Jak se máš?",
        "Magyar": "Szia! Hogy vagy?",
        "Română": "Bună! Ce mai faci?",
        "Українська": "Привіт! Як справи?",
    },
    
    "🎯 특수 케이스": {
        "Emoji": "😊🎉🌟💖🚀",
        "Mixed": "Hello 안녕 你好 こんにちは",
        "Numbers": "1234567890 일이삼사",
        "Symbols": "@#$%^&*()_+-=[]{}",
        "Korean Jamo": "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ",
        "URL": "https://example.com/test",
        "Email": "test@example.com",
    }
}

# 테스트 실행
all_results = []

for category, samples in test_samples.items():
    print(f"\n{category}")
    print("-" * 50)
    
    category_results = []
    
    for lang, text in samples.items():
        result = test_language(text, lang)
        category_results.append(result)
        all_results.append(result)
        
        print(f"{lang:20} TF: {result['tf_accuracy']:5.1%}  Char: {result['char_accuracy']:5.1%}  Loss: {result['loss']:.3f}")
        if result['char_accuracy'] < 0.5:
            print(f"  → Reconstructed: {result['reconstructed'][:30]}...")
    
    # 카테고리 평균
    avg_tf = sum(r['tf_accuracy'] for r in category_results) / len(category_results)
    avg_char = sum(r['char_accuracy'] for r in category_results) / len(category_results)
    print(f"\n카테고리 평균: TF {avg_tf:.1%}, Char {avg_char:.1%}")

# 전체 통계
print("\n" + "="*70)
print("전체 통계")
print("="*70)

# 언어별 그룹 통계
major_langs = ["English", "한국어", "中文简体", "日本語", "Español", "Français", "Deutsch", "Русский", "العربية"]
major_results = [r for r in all_results if r['lang'] in major_langs]
minor_results = [r for r in all_results if r['lang'] not in major_langs and r['lang'] not in ["Emoji", "Mixed", "Numbers", "Symbols", "Korean Jamo", "URL", "Email"]]

if major_results:
    avg_major_tf = sum(r['tf_accuracy'] for r in major_results) / len(major_results)
    avg_major_char = sum(r['char_accuracy'] for r in major_results) / len(major_results)
    print(f"주요 언어 평균: TF {avg_major_tf:.1%}, Char {avg_major_char:.1%}")

if minor_results:
    avg_minor_tf = sum(r['tf_accuracy'] for r in minor_results) / len(minor_results)
    avg_minor_char = sum(r['char_accuracy'] for r in minor_results) / len(minor_results)
    print(f"희소 언어 평균: TF {avg_minor_tf:.1%}, Char {avg_minor_char:.1%}")

# 개별 언어 상세
print("\n개별 언어 성능:")
lang_groups = {
    "완벽 (95%+)": [],
    "우수 (80-95%)": [],
    "양호 (60-80%)": [],
    "개선 필요 (<60%)": []
}

for r in all_results:
    if r['char_accuracy'] >= 0.95:
        lang_groups["완벽 (95%+)"].append(r['lang'])
    elif r['char_accuracy'] >= 0.80:
        lang_groups["우수 (80-95%)"].append(r['lang'])
    elif r['char_accuracy'] >= 0.60:
        lang_groups["양호 (60-80%)"].append(r['lang'])
    else:
        lang_groups["개선 필요 (<60%)"].append(r['lang'])

for group, langs in lang_groups.items():
    if langs:
        print(f"\n{group}: {', '.join(langs[:10])}")
        if len(langs) > 10:
            print(f"  ... 외 {len(langs)-10}개")

# JSON으로 결과 저장
results_json = {
    'epoch': checkpoint['epoch'],
    'loss': float(checkpoint['loss']),
    'languages_tested': len(all_results),
    'average_tf_accuracy': sum(r['tf_accuracy'] for r in all_results) / len(all_results),
    'average_char_accuracy': sum(r['char_accuracy'] for r in all_results) / len(all_results),
    'language_results': all_results
}

with open('multilingual_test_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_json, f, ensure_ascii=False, indent=2)

print(f"\n결과가 multilingual_test_results.json에 저장되었습니다.")
print("="*70)