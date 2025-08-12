# みまもりAI (mimamori-ai)

子ども向けの「やさしい先生UI＋安全フィルタ＋メール通知」なローカルAIフロント。

## 前提

Windows PCで構成することを前提としています。それ以外のOSの場合、以下のファイルを適切なものと差し替える必要があります。
* start_guardian.bat
* nightly.ps1
* deny_ollama.ps1

## セットアップ
```bash
# 1) 依存
pip install -r requirements.txt

# 2) 環境変数
cp .env.example .env
# 値を編集（SMTPなど）

# 3) 起動
uvicorn guardian:app --host 127.0.0.1 --port 8787
# ブラウザで http://127.0.0.1:8787/

## NGワード管理
NGワードは lexicons/ng_words.yaml（公開）＋ lexicons/ng_words.local.yaml（任意・非公開）をマージして使います。