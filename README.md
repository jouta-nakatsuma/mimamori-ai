# みまもりAI (mimamori-ai)

子ども向けの「やさしい先生UI＋安全フィルタ＋メール通知」なローカルAIフロント。

## 前提

* 本プログラムは Windows PC を前提とします（他OSは相当のファイルに置き換えてください）
  * `start_guardian.bat`
  * `nightly.ps1`
  * `deny-ollama.ps1`
* ローカルに Ollama をインストールしていること  
  https://ollama.com/


## セットアップ
```bash
# 0) Python 3.10+ 推奨（3.12/3.13 動作確認）
# 1) 仮想環境（任意だが推奨）
python -m venv .venv && .\.venv\Scripts\activate

# 2) 依存
pip install -r requirements.txt

# 3) 環境変数
copy .env.example .env
# .env を編集（下記のキー一覧を参照）

# 4) 起動
uvicorn guardian:app --host 127.0.0.1 --port 8787
# ブラウザ: http://127.0.0.1:8787/

```
### .env の主なキー
```bash
# メール（必須）
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587           # 465 の場合はSSLで自動切替
SMTP_USER=your@gmail.com
SMTP_PASS=app-password-16chars
SMTP_FROM="みまもりAI <your@gmail.com>"
SMTP_TO=parent@example.com

# 夜間日報のトークン
NIGHTLY_TOKEN=change-me

# 危険出力の二次チェックに使うモデル（任意）
REVIEW_MODEL=gpt-5      # 例: gpt-5, gpt-5-mini, gpt-4o

# OpenAI を使う場合の鍵（REVIEW_MODELを使うなら必要）
OPENAI_API_KEY=sk-xxxx
```
> メールが届かないときは `/debug/env` でキーが読めているか、`/debug/mail` で疎通確認できます。

### Windows の自動起動/日報（任意）
管理者 PowerShell:
```powershell
# PC起動時にサーバ起動（SYSTEM権限）
schtasks /Create /TN "Guardian_OnStart" /SC ONSTART /RU "SYSTEM" /TR "C:\Users\sunpi\ai\childai\start_guardian.bat" /RL HIGHEST /F

# 毎日21:30に日報送信
schtasks /Create /TN "Guardian_Nightly" /SC DAILY /ST 21:30 /RU "SYSTEM" `
  /TR "powershell -NoProfile -Command \"Invoke-RestMethod -Method POST -Uri 'http://127.0.0.1:8787/cron/nightly?token=YOUR_TOKEN' | Out-Null\"" /F

```
> PowerShell の `curl` は Invoke-WebRequest の別名です。テストには `Invoke-RestMethod` か `curl.exe` を使ってください。

## NGワード管理
- 公開辞書: `lexicons/ng_words.yaml`
- 家庭ごとの上書き: `lexicons/ng_words.local.yaml`（**コミットしない**）

両方が存在する場合はマージして判定します（`local` が優先）。公開リポには `ng_words.yaml` のみを含め、強めの語は `ng_words.local.yaml` に置く運用を推奨。

## ログと日報
- 入力のみ: `logs/prompts.jsonl`
- 入出力（AIの raw / shown 含む）: `logs/turns.jsonl`
- 日報はメールで送信、**整形テキスト `transcript-YYYY-MM-DD.txt`** を添付（jsonlは添付しません）

### デバッグ
- `GET /debug/env` … .env の読込状況（値そのものは出しません）
- `GET /debug/mail` … テストメール送信

## ディレクトリ
```csharp
.
├─ guardian.py
├─ static/
│  └─ index.html
├─ prompts/
│  └─ teacher.txt
├─ lexicons/
│  ├─ ng_words.yaml
│  └─ ng_words.local.yaml   # 任意・非公開
├─ logs/                    # 実行時作成・非公開
├─ start_guardian.bat
├─ nightly.ps1
├─ requirements.txt
├─ .env.example
└─ LICENSE
```

## License
MIT © 2025 Jouta Nakatsuma