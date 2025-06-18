# main.py (修正後)

from fastapi import (
    FastAPI, APIRouter, HTTPException, Body, Depends, status, Request
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Any, Dict
from datetime import datetime
import logging
from enum import Enum
import MetaTrader5 as mt5  # --- 修正点: mt5定数を使用するためにインポート ---

# --- 外部モジュールのインポート (mt5_handler.py は別途必要) ---
# このファイルと同じディレクトリに mt5_handler.py が存在することを想定
from mt5_handler import MT5Handler, MT5ConnectionError, TradeExecutionError

# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 設定管理 ---
class Settings(BaseSettings):
    """環境変数から設定を読み込むクラス"""
    mt5_login: int
    mt5_password: str
    mt5_server: str
    mt5_path: str

    class Config:
        env_file = ".env"

# --- PydanticモデルとEnum定義 ---

# --- 修正点: API仕様(integer)とMT5ライブラリの定数に合わせる ---
class OrderAction(int, Enum):
    """取引アクション。MT5の定数に対応する整数値を使用。"""
    DEAL = mt5.TRADE_ACTION_DEAL
    PENDING = mt5.TRADE_ACTION_PENDING
    SLTP = mt5.TRADE_ACTION_SLTP
    MODIFY = mt5.TRADE_ACTION_MODIFY
    REMOVE = mt5.TRADE_ACTION_REMOVE

class OrderType(int, Enum):
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5
    BUY_STOP_LIMIT = 6
    SELL_STOP_LIMIT = 7

class OrderTimeType(int, Enum):
    GTC = 0
    DAY = 1
    SPECIFIED = 2
    SPECIFIED_DAY = 3

# --- 修正点: filling typeの定数値をMT5に合わせる ---
# 元のコードではFOK=1, IOC=2, RETURN=3でしたが、MT5ではFOK=0, IOC=1, RETURN=2が一般的です。
# ただし、mt5_handler側で自動設定されるため、クライアントからの指定は任意です。
# mt5_handlerの_get_filling_modeに合わせて修正します。
class OrderFillingType(int, Enum):
    FOK = mt5.ORDER_FILLING_FOK
    IOC = mt5.ORDER_FILLING_IOC
    RETURN = mt5.ORDER_FILLING_RETURN

class AccountInfoResponse(BaseModel):
    login: int
    balance: float
    currency: str

class HistoricalDataRequest(BaseModel):
    symbol: str = Field(..., example="EURUSD", description="通貨ペア名")
    timeframe: str = Field(..., example="H1", description="時間足 (e.g., M1, M5, H1, D1)")
    count: int = Field(100, gt=0, le=5000, description="取得するバーの数")
    from_date: Optional[datetime] = Field(None, description="取得開始日時 (ISO 8601形式)")
    to_date: Optional[datetime] = Field(None, description="取得終了日時 (ISO 8601形式)")

class TradeRequest(BaseModel):
    action: OrderAction = Field(..., description="取引アクション")
    symbol: str = Field(..., example="EURUSD", description="通貨ペア名")
    volume: float = Field(..., gt=0, example=0.01, description="ロット数")
    type: OrderType = Field(..., description="注文タイプ")
    price: Optional[float] = Field(None, example=1.12345, description="注文価格（指値等）")
    sl: Optional[float] = Field(None, example=1.12000, description="ストップロス価格")
    tp: Optional[float] = Field(None, example=1.13000, description="テイクプロフィット価格")
    deviation: Optional[int] = Field(20, example=20, description="許容スリッページ")
    magic: Optional[int] = Field(234000, example=234000, description="マジックナンバー")
    comment: Optional[str] = Field("python script open", description="コメント")
    type_time: Optional[OrderTimeType] = Field(None, description="注文有効期限タイプ")
    type_filling: Optional[OrderFillingType] = Field(None, description="注文執行条件")
    position: Optional[int] = Field(None, description="決済または変更対象のポジションチケットID")
    order: Optional[int] = Field(None, description="変更または削除対象のオーダーチケットID")
    stoplimit: Optional[float] = Field(None, example=1.12350, description="ストップリミット注文の価格")
    expiration: Optional[int] = Field(None, description="有効期限（type_timeがSPECIFIEDの場合）")

class ClosePositionRequest(BaseModel):
    volume: Optional[float] = Field(None, gt=0, description="部分決済する場合のロット数。Noneの場合は全決済。")
    deviation: int = Field(20, description="許容スリッページ")

class ModifyPositionRequest(BaseModel):
    sl: Optional[float] = Field(None, description="新しいストップロス価格")
    tp: Optional[float] = Field(None, description="新しいテイクプロフィット価格")

# --- アプリケーションインスタンスとMT5ハンドラ ---
mt5_handler: Optional[MT5Handler] = None

app = FastAPI(
    title="MetaTrader 5 Bridge API",
    description="An API to control a local MetaTrader 5 terminal for algorithmic trading.",
    version="1.0.0"
)

# --- ライフサイクルイベント ---
@app.on_event("startup")
def startup_event():
    """アプリケーション起動時にMT5に接続"""
    global mt5_handler
    try:
        settings = Settings()
        # MT5Handlerの初期化時に接続とログインが行われる
        mt5_handler = MT5Handler(
            login=settings.mt5_login,
            password=settings.mt5_password,
            server=settings.mt5_server,
            path=settings.mt5_path
        )
        logging.info("Successfully initialized and connected to MetaTrader 5.")
    except MT5ConnectionError as e:
        logging.critical(f"Failed to initialize MT5 handler during startup: {e}")
        mt5_handler = None
    except Exception as e:
        logging.critical(f"An unexpected error occurred during startup: {e}")
        mt5_handler = None


@app.on_event("shutdown")
def shutdown_event():
    """アプリケーション終了時にMT5から切断"""
    if mt5_handler:
        mt5_handler.shutdown()
        logging.info("Successfully disconnected from MetaTrader 5.")

# --- 依存性注入 ---
def get_mt5_handler() -> MT5Handler:
    """MT5ハンドラインスタンスを取得する依存性注入関数"""
    if mt5_handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not connect to MetaTrader 5. Check server logs for details."
        )
    return mt5_handler

# --- エラーハンドリング ---
@app.exception_handler(TradeExecutionError)
async def trade_execution_exception_handler(request: Request, exc: TradeExecutionError):
    logging.error(f"Trade Execution Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)}
    )

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logging.error(f"Value Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

@app.exception_handler(MT5ConnectionError)
async def mt5_connection_exception_handler(request: Request, exc: MT5ConnectionError):
    logging.error(f"MT5 Connection Error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"MetaTrader 5 connection error: {exc}"}
    )

# --- APIRouter の定義 ---
router_status = APIRouter(prefix="/status", tags=["Status"])
router_account = APIRouter(prefix="/account", tags=["Account"])
router_market = APIRouter(prefix="/market", tags=["Market Data"])
router_trading = APIRouter(prefix="/trading", tags=["Trading"])


# --- ステータスエンドポイント ---
@router_status.get("/ping", summary="API疎通確認")
def ping():
    return {"status": "pong"}

@router_status.get("/mt5", summary="MT5ターミナル接続確認")
def get_mt5_status(handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_terminal_info()

# --- 口座情報エンドポイント ---
@router_account.get("/", response_model=Dict, summary="口座詳細情報取得")
def get_account_info(handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_account_info()

# --- マーケットデータエンドポイント ---
@router_market.get("/symbols/tradable", response_model=List[str], summary="取引可能シンボルリスト取得")
def get_tradable_symbols(handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_tradable_symbols()

@router_market.get("/symbols/all", response_model=List[Dict[str, Any]], summary="全シンボル詳細情報取得")
def get_all_symbol_details(handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_all_symbol_details()

@router_market.get("/symbols/{symbol_name}", response_model=Dict[str, Any], summary="特定シンボル詳細情報取得")
def get_symbol_info(symbol_name: str, handler: MT5Handler = Depends(get_mt5_handler)):
    info = handler.get_symbol_info(symbol_name.upper())
    if info is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol_name}' not found.")
    return info

@router_market.post("/history", response_model=List[Dict[str, Any]], summary="ヒストリカルデータ取得")
def get_historical_data(request: HistoricalDataRequest, handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_historical_data(
        symbol=request.symbol,
        timeframe_str=request.timeframe,
        count=request.count,
        from_date=request.from_date,
        to_date=request.to_date
    )

# --- 取引エンドポイント ---
@router_trading.post("/orders", status_code=status.HTTP_201_CREATED, summary="新規注文（成行・指値）")
def place_order(request: TradeRequest, handler: MT5Handler = Depends(get_mt5_handler)):
    # --- 修正点: 冗長な代入を削除し、model_dumpの結果をそのまま渡す ---
    # model_dump()がEnumを自動的に正しい整数値に変換してくれる
    request_dict = request.model_dump(exclude_none=True)
    return handler.place_order(request_dict)

@router_trading.get("/positions", response_model=List[Dict[str, Any]], summary="保有ポジション一覧取得")
def get_open_positions(symbol: Optional[str] = None, handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_open_positions(symbol=symbol.upper() if symbol else None)

@router_trading.get("/positions/{ticket}", response_model=Dict[str, Any], summary="特定ポジション取得")
def get_position_by_ticket(ticket: int, handler: MT5Handler = Depends(get_mt5_handler)):
    # --- 修正点: 存在しないメソッド呼び出しを修正 ---
    position_list = handler.get_open_positions(ticket=ticket)
    if not position_list:
        raise HTTPException(status_code=404, detail=f"Position with ticket {ticket} not found.")
    return position_list[0]

@router_trading.delete("/positions/{ticket}", status_code=status.HTTP_200_OK, summary="ポジション決済")
def close_position(ticket: int, request: ClosePositionRequest, handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.close_position(ticket=ticket, volume=request.volume, deviation=request.deviation)

@router_trading.patch("/positions/{ticket}", status_code=status.HTTP_200_OK, summary="ポジション変更 (SL/TP)")
def modify_position(ticket: int, request: ModifyPositionRequest, handler: MT5Handler = Depends(get_mt5_handler)):
    if request.sl is None and request.tp is None:
        raise HTTPException(status_code=400, detail="Either 'sl' or 'tp' must be provided.")
    return handler.modify_position_sltp(ticket=ticket, sl=request.sl, tp=request.tp)

@router_trading.get("/orders", response_model=List[Dict[str, Any]], summary="待機注文一覧取得")
def get_pending_orders(symbol: Optional[str] = None, handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.get_pending_orders(symbol=symbol.upper() if symbol else None)

@router_trading.get("/orders/{ticket}", response_model=Dict[str, Any], summary="特定待機注文取得")
def get_order_by_ticket(ticket: int, handler: MT5Handler = Depends(get_mt5_handler)):
    # --- 修正点: 存在しないメソッド呼び出しを修正 ---
    order_list = handler.get_pending_orders(ticket=ticket)
    if not order_list:
        raise HTTPException(status_code=404, detail=f"Pending order with ticket {ticket} not found.")
    return order_list[0]

@router_trading.delete("/orders/{ticket}", status_code=status.HTTP_200_OK, summary="待機注文キャンセル")
def cancel_pending_order(ticket: int, handler: MT5Handler = Depends(get_mt5_handler)):
    return handler.cancel_pending_order(ticket=ticket)

# --- ルーターをアプリケーションに登録 ---
app.include_router(router_status)
app.include_router(router_account)
app.include_router(router_market)
app.include_router(router_trading)

# --- サーバー起動 ---
if __name__ == "__main__":
    import uvicorn
    # 起動するファイル名(モジュール名)を "main" に修正
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)