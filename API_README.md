Part 2: API操作マニュアル (README.md)

このセクションは、AIエージェントの開発者がブリッジAPIをセットアップし、利用するための完全なガイドです。マークダウン形式で提供されます。
# MetaTrader 5 Bridge API
これは、ローカルのWindows環境で動作するMetaTrader 5（MT5）ターミナルを、REST API経由で操作するためのブリッジサーバーです。WSL（Windows Subsystem for Linux）などで動作する外部のAIトレーディングエージェントから、MT5のほぼ全ての機能を制御することを目的としています。

## 1. システム要件

- **OS**: Windows 10 / 11
- **Python**: 3.8以上
- **MetaTrader 5**: デスクトップ版ターミナルがインストールされ、取引口座にログイン済みであること。
- **依存ライブラリ**: fastapi, uvicorn, python-dotenv, pandas, pytz, MetaTrader5

## 2. インストールと設定


### ステップ 1: Python環境の構築

コマンドプロンプトまたはPowerShellを開き、プロジェクトディレクトリで以下のコマンドを実行します。

```bash
# 仮想環境の作成
python -m venv venv

# 仮想環境のアクティベート
.\venv\Scripts\activate

# 必要なライブラリのインストール
pip install fastapi uvicorn "uvicorn[standard]" python-dotenv pandas pytz MetaTrader5
```



### ステップ 2: MetaTrader 5 ターミナルの設定

このステップは非常に重要です。設定を誤るとAPIは機能しません。

1.  MT5ターミナルを開き、メニューバーから **「ツール」 -> 「オプション」** を選択します。
2.  開いたウィンドウで **「エキスパートアドバイザ」** タブを選択します。
3.  **「アルゴリズム取引を許可」** のチェックボックスをオンにします [3]。
4.  **「DLLの使用を許可する（信頼できるアプリケーションのみ）」** のチェックボックスをオンにします [3]。
5.  OKをクリックしてウィンドウを閉じます。
6.  最後に、MT5ターミナルのツールバーにある **「アルゴリズム取引」** ボタンをクリックして、アイコンが緑色の再生マークになることを確認します。

![MT5設定画面](https://i.imgur.com/your-image-placeholder.png)

### ステップ 3: APIサーバーの設定

プロジェクトのルートディレクトリに `.env` という名前のファイルを作成し、以下の内容を記述します。自身のMT5の情報を入力してください。

```dotenv
#.env file
MT5_LOGIN=12345678
MT5_PASSWORD="your_password"
MT5_SERVER="YourBroker-Server"
# MT5ターミナルのterminal64.exeへのフルパス
MT5_PATH="C:\Program Files\MetaTrader 5\terminal64.exe"
```



### ステップ 4: APIサーバーの起動

設定が完了したら、仮想環境がアクティブな状態で以下のコマンドを実行してサーバーを起動します。

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```


--host 0.0.0.0 を指定することで、ローカルネットワーク内の他のデバイス（WSLを含む）からアクセスできるようになります。

## 3. ネットワーク通信：WSLからWindowsホストへの接続

WSL環境で動作するAIエージェントから、Windowsホスト上で実行されているこのAPIサーバーに接続するには、特別な注意が必要です。
WSL2は独自の仮想ネットワーク上で動作するため、WSL内の localhost はWindowsホストの localhost とは異なります。WSLからホストにアクセスするには、ホストのIPアドレスを使用する必要があります。
ホストIPの特定: Windowsのコマンドプロンプトで ipconfig を実行し、「イーサネット アダプター vEthernet (WSL)」のアダプター情報に表示されるIPv4アドレスを探します。これがWSLから見たホストのIPアドレスです。
APIへのアクセス: WSL内のクライアント（Pythonスクリプトなど）から、http://<ホストのIPアドレス>:8000 のようにしてAPIにアクセスします。
注意: Windows Defender ファイアウォール
デフォルトでは、Windows Defender ファイアウォールが外部からのポート8000への接続をブロックする可能性があります。接続できない場合は、ポート8000に対する受信規則を新たに追加する必要があります。

## 4. APIエンドポイントリファレンス

APIは論理的なグループに分かれています。

| HTTP Method | Path | Function | Brief Description |
|-------------|------|----------|-------------------|
| GET | /status/ping | Ping | サーバーの生存確認をします。 |
| GET | /status/mt5 | MT5 Status | MT5ターミナルへの接続状態を確認します。 |
| GET | /account/ | Account Information | 取引口座の詳細情報を取得します。 |
| GET | /market/symbols/tradable | Get Tradable Symbols | 取引可能な全シンボルのリストを取得します。 |
| GET | /market/symbols/all | Get All Symbol Details | 利用可能な全シンボルの詳細情報を取得します。 |
| GET | /market/symbols/{symbol_name} | Get Symbol Info | 特定のシンボルの詳細情報を取得します。 |
| POST | /market/history | Get Historical Data | シンボルのヒストリカルデータを取得します。 |
| POST | /trading/orders | Place Order | 新規の成行注文または待機注文を発注します。 |
| GET | /trading/orders | Get Pending Orders | 全ての待機注文を取得します。 |
| GET | /trading/orders/{ticket} | Get Pending Order by Ticket | 特定の待機注文を取得します。 |
| DELETE | /trading/orders/{ticket} | Cancel Pending Order | 待機注文をキャンセルします。 |
| GET | /trading/positions | Get Open Positions | 全てのオープンポジションを取得します。 |
| GET | /trading/positions/{ticket} | Get Position by Ticket | 特定のオープンポジションを取得します。 |
| DELETE | /trading/positions/{ticket} | Close Position | オープンポジションを決済します。 |
| PATCH | /trading/positions/{ticket} | Modify Position SL/TP | ポジションのSL/TPを変更します。 |


### 4.1. Status Endpoints (/status)


#### GET /status/ping

サーバーが稼働しているかを確認するためのシンプルなエンドポイントです。

**Response (200 OK):**
```json
{
  "status": "pong"
}
```



#### GET /status/mt5

MT5ターミナルへの接続状態を確認し、ターミナルの基本情報を返します。

**Response (200 OK):**
```json
{
  "community_account": false,
  "community_connection": false,
  "connected": true,
  "dlls_allowed": true,
  "trade_allowed": true,
  "email_enabled": false,
  "ftp_enabled": false,
  "notifications_enabled": false,
  "mqid": false,
  "build": 3802,
  "maxbars": 100000,
  "codepage": 932,
  "ping_last": 123456,
  "community_balance": 0.0,
  "retransmission": 0.0,
  "company": "MetaQuotes Software Corp.",
  "name": "MetaTrader 5",
  "path": "C:\\Program Files\\MetaTrader 5"
}
```


**Error Response (503 Service Unavailable):** MT5に接続できない場合。
```json
{
  "detail": "Could not connect to MetaTrader 5:..."
}
```



### 4.2. Market Data Endpoints (/market)


#### GET /market/symbols/tradable

取引が完全に許可されている（SYMBOL_TRADE_MODE_FULL）シンボルの名前のリストを返します。AIエージェントが取引対象を決定する際の最初のステップとして有用です。

**Response (200 OK):**
```json
[
  "EURUSD",
  "USDJPY",
  "GBPUSD"
]
```



#### POST /market/history

指定されたシンボルのヒストリカルデータを取得します。

**Request Body:**
```json
{
  "symbol": "EURUSD",
  "timeframe": "H1",
  "count": 10
}
```


**Timeframe Constants:**
| API String | Description |
|------------|------------------------------|
| M1 | 1分足 |
| M5 | 5分足 |
| M15 | 15分足 |
| M30 | 30分足 |
| H1 | 1時間足 |
| H4 | 4時間足 |
| D1 | 日足 |
| W1 | 週足 |
| MN1 | 月足 |

**Response (200 OK):**
```json
[
  {
    "time": "2023-12-01T10:00:00",
    "open": 1.0565,
    "high": 1.0570,
    "low": 1.0560,
    "close": 1.0568,
    "tick_volume": 1250,
    "spread": 2,
    "real_volume": 0
  }
]
```



### 4.3. Trading Endpoints (/trading)

これらのエンドポイントは、MQL5の取引モデルを反映しており、「待機注文（Orders）」と「オープンポジション（Positions）」を明確に区別します 5。

#### POST /trading/orders

新規の注文を発注します。リクエストボディのactionフィールドによって、成行注文か待機注文かが決まります。

**Trade Request Body Fields:**
| Field | Type | Description |
|--------------|---------|---------------------------------------------------------------------------------------------------------|
| action | integer | 1 (DEAL: 成行), 5 (PENDING: 待機) |
| symbol | string | 取引シンボル (例: "EURUSD") |
| volume | float | ロット数 (例: 0.1) |
| type | integer | 注文タイプ (0: BUY, 1: SELL, 2: BUY_LIMIT, etc.) |
| price | float | 注文価格 (成行の場合は不要、待機注文では必須) |
| sl | float | ストップロス価格 (任意) |
| tp | float | テイクプロフィット価格 (任意) |
| position | integer | 決済または変更対象のポジションチケット (決済時に使用) |
| order | integer | 変更または削除対象の待機注文チケット |
| type_filling| integer | 注文執行方針 (0: FOK, 1: IOC, 2: RETURN)。APIが自動で最適なものを選択するため通常は不要 |
**Example: Market Buy Order**
```json
{
  "action": 1,
  "symbol": "EURUSD",
  "volume": 0.01,
  "type": 0,
  "comment": "AI agent market buy"
}
```


**Example: Buy Limit Order**
```json
{
  "action": 5,
  "symbol": "USDJPY",
  "volume": 0.02,
  "type": 2,
  "price": 149.50,
  "sl": 149.20,
  "tp": 150.50,
  "comment": "AI agent buy limit"
}
```


**Response (201 Created):** 成功した取引結果
```json
{
  "retcode": 10009,
  "deal": 123456789,
  "order": 987654321,
  "volume": 0.01,
  "price": 1.0565,
  "bid": 1.05648,
  "ask": 1.05652,
  "comment": "Request executed",
  "request_id": 1,
  "retcode_external": 0,
  "request": {}
}
```


**Error Response (422 Unprocessable Entity):** 取引チェックまたは送信に失敗した場合。detailにMT5からのエラーメッセージが含まれます。
```json
{
  "detail": "Order check failed: invalid stops"
}
```



#### DELETE /trading/positions/{ticket}

オープンポジションを決済します。

**Request Body:**
```json
{
  "volume": 0.01, // 任意。指定すると部分決済。なければ全量決済。
  "deviation": 20
}
```



#### PATCH /trading/positions/{ticket}

オープンポジションのStop Loss (SL) / Take Profit (TP) を変更します。

**Request Body:**
```json
{
  "sl": 1.12500,
  "tp": 1.13500
}
```



#### DELETE /trading/orders/{ticket}

待機注文をキャンセルします。

## 5. エラーハンドリング

APIは標準的なHTTPステータスコードを使用して結果を伝えます。

- **200 OK / 201 Created**: リクエスト成功。
- **400 Bad Request**: リクエストの形式が不正（例: 必須フィールドの欠落）。
- **404 Not Found**: 指定されたリソース（シンボル、チケット等）が見つからない。
- **422 Unprocessable Entity**: リクエストの形式は正しいが、取引ロジック上の問題で処理できない（例: 証拠金不足、無効なSL/TP）。
- **503 Service Unavailable**: MT5ターミナルに接続できない。

## 6. 高度な利用例とワークフロー


### ワークフロー 1: 完全な取引ライフサイクル

1. **口座情報確認**: GET /account/ を呼び出し、margin_free を確認。
2. **待機注文発注**: POST /trading/orders でBuy Stop注文を発注。
3. **注文確認**: GET /trading/orders を定期的に呼び出し、注文がリストにあることを確認。
4. **ポジション化確認**: 注文が約定すると /trading/orders から消え、GET /trading/positions のリストに現れる。
5. **SL/TP変更**: PATCH /trading/positions/{ticket} でトレーリングストップなどを実装。
6. **部分決済**: DELETE /trading/positions/{ticket} で volume を指定して利益の一部を確定。
7. **完全決済**: DELETE /trading/positions/{ticket} で残りのポジションを決済。

### ワークフロー 2: 市場分析データの収集

1. **取引対象の選定**: GET /market/symbols/tradable で取引可能なシンボルリストを取得。
2. **シンボル詳細の取得**: GET /market/symbols/EURUSD を呼び出し、point（最小価格変動）、volume_step（ロット数の刻み）などの詳細情報を取得。
3. **ヒストリカルデータ取得**: POST /market/history で日足（D1）の過去1000本分データを取得し、AIモデルの学習データとする。

### Symbol Trade Mode Reference

GET /market/symbols/{symbol_name} で返される trade_mode_description フィールドは、AIエージェントのリスク管理において極めて重要です。

| API Status String | MQL5 Constant | Meaning for Trading Agent |
|-------------------|---------------|---------------------------|
| FULL | SYMBOL_TRADE_MODE_FULL | 制限なし。新規・決済ともに可能。 |
| CLOSE_ONLY | SYMBOL_TRADE_MODE_CLOSEONLY | 新規注文は不可。既存ポジションの決済のみ可能。 |
| LONG_ONLY | SYMBOL_TRADE_MODE_LONGONLY | 買いポジションの保有と決済のみ可能。売りは不可。 |
| SHORT_ONLY | SYMBOL_TRADE_MODE_SHORTONLY | 売りポジションの保有と決済のみ可能。買いは不可。 |
| DISABLED | SYMBOL_TRADE_MODE_DISABLED | このシンボルでの取引は完全に無効化されている。 |



