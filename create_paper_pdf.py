#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create PDF from markdown paper with academic formatting
"""

import sys
import os

# UTF-8 출력 설정
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("PDF 생성 옵션들:")
print("\n1. Markdown → PDF (pandoc 사용)")
print("   - 설치: https://pandoc.org/installing.html")
print("   - 명령어: pandoc paper_english.md -o paper.pdf --pdf-engine=xelatex")
print("   - LaTeX 템플릿 사용 가능")

print("\n2. Markdown → LaTeX → PDF (학술 논문 양식)")
print("   - IEEE/ACL/NeurIPS 템플릿 사용 가능")
print("   - 더 전문적인 레이아웃")

print("\n3. Python 라이브러리 사용:")
print("   - pip install markdown2pdf")
print("   - pip install weasyprint")
print("   - pip install reportlab")

print("\n4. 온라인 변환 (즉시 가능):")
print("   - https://www.markdowntopdf.com/")
print("   - https://md2pdf.netlify.app/")
print("   - Typora, Obsidian 등 마크다운 에디터 사용")

print("\n추천 방법:")
print("1. 빠른 PDF: 온라인 변환기 사용")
print("2. 전문적인 PDF: LaTeX 템플릿 사용")

# 간단한 HTML 변환 (weasyprint 필요)
try:
    import markdown
    import weasyprint
    
    def create_simple_pdf():
        # Markdown to HTML
        with open('paper_english.md', 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content)
        
        # Add CSS for academic styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Times New Roman', serif;
                    font-size: 11pt;
                    line-height: 1.5;
                    margin: 2.5cm;
                    text-align: justify;
                }}
                h1 {{
                    font-size: 16pt;
                    text-align: center;
                    margin-bottom: 1em;
                }}
                h2 {{
                    font-size: 12pt;
                    font-weight: bold;
                    margin-top: 1em;
                }}
                h3 {{
                    font-size: 11pt;
                    font-weight: bold;
                    font-style: italic;
                }}
                code {{
                    font-family: 'Courier New', monospace;
                    background-color: #f5f5f5;
                    padding: 2px 4px;
                }}
                table {{
                    border-collapse: collapse;
                    margin: 1em auto;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 5px 10px;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        # Save HTML
        with open('paper.html', 'w', encoding='utf-8') as f:
            f.write(styled_html)
        
        # Convert to PDF
        weasyprint.HTML(string=styled_html).write_pdf('paper_simple.pdf')
        print("✅ paper_simple.pdf 생성됨")
        
except ImportError:
    print("\nweasyprint 설치 필요: pip install weasyprint markdown")
    print("또는 온라인 변환기를 사용하세요")