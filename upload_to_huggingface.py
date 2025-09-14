#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hugging Face Hub 자동 업로드 스크립트
"""

import os
import sys
import torch
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file, upload_folder

# 상위 경로 추가
sys.path.append(str(Path(__file__).parent.parent / "intelligent-tokenizer_v6.0"))

def upload_to_huggingface(
    repo_name="intelligent-tokenizer-v6",
    organization=None,  # None이면 개인 계정
    private=False
):
    """Hugging Face Hub에 모델 업로드"""
    
    print("🚀 Hugging Face Hub 업로드 시작...")
    
    # 1. API 초기화
    api = HfApi()
    
    # 2. Repository 이름 설정
    if organization:
        repo_id = f"{organization}/{repo_name}"
    else:
        repo_id = repo_name  # 개인 계정은 username이 자동으로 붙음
    
    try:
        # 3. Repository 생성
        print(f"📦 Creating repository: {repo_id}")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print("✅ Repository created/verified")
        
        # 4. 체크포인트를 PyTorch 형식으로 변환
        checkpoint_path = Path("../intelligent-tokenizer_v6.0/checkpoints/unified/latest_checkpoint.pt")
        if checkpoint_path.exists():
            print("🔄 Converting checkpoint to Hugging Face format...")
            
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            
            # 모델 state_dict만 저장
            torch.save(
                checkpoint['model_state_dict'],
                "pytorch_model.bin"
            )
            print("✅ pytorch_model.bin created")
            
            # Training state 저장 (선택사항)
            training_state = {
                "epoch": checkpoint['epoch'],
                "loss": checkpoint['loss'],
                "model_config": checkpoint['model_config']
            }
            torch.save(training_state, "training_state.bin")
            print("✅ training_state.bin created")
        
        # 5. 파일 업로드
        files_to_upload = [
            "README.md",
            "config.json",
            "tokenizer_config.json",
            "pytorch_model.bin",
            "training_state.bin",
            "requirements.txt",
            ".gitattributes"
        ]
        
        print("\n📤 Uploading files...")
        for file in files_to_upload:
            if Path(file).exists():
                print(f"  Uploading {file}...")
                upload_file(
                    path_or_fileobj=file,
                    path_in_repo=file,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ {file} uploaded")
            else:
                print(f"  ⚠️ {file} not found, skipping...")
        
        # 6. 소스 코드 업로드 (선택사항)
        src_files = {
            "../intelligent-tokenizer_v6.0/src/core/byte_tokenizer_v6.py": "byte_tokenizer_v6.py",
            "../intelligent-tokenizer_v6.0/core/boundary_aware_model.py": "boundary_aware_model.py",
        }
        
        print("\n📤 Uploading source code...")
        for src, dst in src_files.items():
            src_path = Path(src)
            if src_path.exists():
                print(f"  Uploading {dst}...")
                upload_file(
                    path_or_fileobj=str(src_path),
                    path_in_repo=dst,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"  ✅ {dst} uploaded")
        
        print("\n" + "="*60)
        print("🎉 업로드 완료!")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")
        print("="*60)
        
        # 7. 사용 방법 출력
        print("\n📖 사용 방법:")
        print("```python")
        print(f"from transformers import AutoModel")
        print(f"model = AutoModel.from_pretrained('{repo_id}', trust_remote_code=True)")
        print("```")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 해결 방법:")
        print("1. huggingface-cli login 실행")
        print("2. https://huggingface.co/settings/tokens 에서 토큰 생성")
        print("3. Write 권한 필요")

if __name__ == "__main__":
    # 먼저 구조 준비
    print("📁 Preparing Hugging Face structure...")
    os.system("python prepare_huggingface.py")
    
    print("\n" + "="*60)
    print("⚠️ 업로드 전 확인사항:")
    print("1. Hugging Face 계정이 있나요?")
    print("2. huggingface-cli login 했나요?")
    print("3. 공개(public) 할건가요? 비공개(private)?")
    print("="*60)
    
    # 사용자 입력
    response = input("\n계속하시겠습니까? (y/n): ")
    if response.lower() == 'y':
        username = input("Hugging Face username (또는 organization): ")
        private = input("Private repository? (y/n): ").lower() == 'y'
        
        upload_to_huggingface(
            repo_name="intelligent-tokenizer-v6",
            organization=username if username else None,
            private=private
        )
    else:
        print("📝 수동 업로드 방법:")
        print("1. https://huggingface.co/new 에서 모델 생성")
        print("2. git clone https://huggingface.co/username/model-name")
        print("3. 파일 복사 후 git push")