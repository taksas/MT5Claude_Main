# mt5_handler.py (修正済み)

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import pytz
import logging
from typing import List, Dict, Any, Optional, Literal

# --- 変更点: 型ヒントをより具体的に ---
# Dict[str, Any] のエイリアスを定義して可読性を向上
SymbolInfo = Dict[str, Any]
AccountInfo = Dict[str, Any]
TerminalInfo = Dict[str, Any]
HistoricalData = List[Dict[str, Any]]
PositionInfo = Dict[str, Any]
OrderInfo = Dict[str, Any]
TradeResult = Dict[str, Any]


# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# カスタム例外
class MT5ConnectionError(Exception):
    """MT5への接続またはログインに失敗した場合の例外"""
    pass

class TradeExecutionError(Exception):
    """取引の実行（チェックまたは送信）に失敗した場合の例外"""
    pass

class MT5Handler:
    """
    MetaTrader 5との全ての対話を管理するクラス。
    接続状態の管理、データ取得、取引実行をカプセル化する。
    
    --- 改善点: コンテキストマネージャ対応 ---
    'with'ステートメントでインスタンスを管理でき、ブロックを抜ける際に
    自動的にshutdown()が呼び出されるため、リソース管理が安全になります。
    
    使用例:
    with MT5Handler(login, password, server, path) as mt5_handler:
        print(mt5_handler.get_account_info())
    """
    
    # --- 改善点: 定数をクラス変数として定義 ---
    # メソッド呼び出しの度に辞書が生成されるのを防ぎ、コードを整理します。
    _TIMEFRAME_MAP = {
        "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6, "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
    }

    _TRADE_MODE_MAP = {
        mt5.SYMBOL_TRADE_MODE_DISABLED: 'DISABLED',
        mt5.SYMBOL_TRADE_MODE_LONGONLY: 'LONG_ONLY',
        mt5.SYMBOL_TRADE_MODE_SHORTONLY: 'SHORT_ONLY',
        mt5.SYMBOL_TRADE_MODE_CLOSEONLY: 'CLOSE_ONLY',
        mt5.SYMBOL_TRADE_MODE_FULL: 'FULL',
    }

    def __init__(self, login: int, password: str, server: str, path: str):
        self._login = login
        self._password = password
        self._server = server
        self._path = path
        self._initialize()

    def _initialize(self):
        """MT5ターミナルへの接続とログインを試みる"""
        # --- 改善点: 初期化とログイン検証の堅牢化 ---
        # initializeでログイン情報を提供し、account_infoでログイン状態を検証します。
        # これにより、不要なlogin()呼び出しをなくし、ロジックをシンプルにします。
        if not mt5.initialize(path=self._path, login=self._login, password=self._password, server=self._server):
            error_code = mt5.last_error()
            logging.error(f"MT5 initialize() failed, error code = {error_code}")
            mt5.shutdown()
            raise MT5ConnectionError(f"Failed to initialize MT5 terminal. Error: {error_code}")

        account_info = mt5.account_info()
        if account_info is None or account_info.login != self._login:
            error_code = mt5.last_error()
            logging.error(f"Failed to login to account {self._login}. Error: {error_code}")
            mt5.shutdown()
            raise MT5ConnectionError(f"Failed to login to account {self._login}. Error: {error_code}")
        
        logging.info(f"Logged in to account {self._login} successfully.")

    def shutdown(self):
        """MT5ターミナルとの接続をシャットダウンする"""
        mt5.shutdown()
        logging.info("MT5 connection shut down.")
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def get_account_info(self) -> AccountInfo:
        """現在の取引口座情報を取得する"""
        info = mt5.account_info()
        if info is None:
            raise MT5ConnectionError("Could not get account info. Connection may be lost.")
        return info._asdict()

    def get_terminal_info(self) -> TerminalInfo:
        """接続されているMT5ターミナルの状態とパラメータを取得する"""
        info = mt5.terminal_info()
        if info is None:
            raise MT5ConnectionError("Could not get terminal info. Connection may be lost.")
        return info._asdict()

    def get_all_symbol_details(self) -> List[SymbolInfo]:
        """ブローカーから利用可能なすべての金融商品の詳細情報を取得する"""
        symbols = mt5.symbols_get()
        # --- 改善点: 戻り値の一貫性 ---
        # データがない場合はNoneではなく空リストを返すことで、呼び出し側の処理を簡潔にします。
        if symbols is None:
            return []
        
        symbols_list = []
        for s in symbols:
            info = s._asdict()
            # --- 改善点: 辞書マッピングによるコードの簡略化 ---
            # if/elifの連鎖を辞書検索に置き換えることで、可読性と保守性が向上します。
            info['trade_mode_description'] = self._TRADE_MODE_MAP.get(info.get('trade_mode'), 'UNKNOWN')
            symbols_list.append(info)
            
        return symbols_list

    def get_tradable_symbols(self) -> List[str]:
        """取引が完全に許可されている（SYMBOL_TRADE_MODE_FULL）金融商品のリストを取得する"""
        all_symbols = self.get_all_symbol_details()
        return [
            s['name'] for s in all_symbols 
            if s.get('trade_mode') == mt5.SYMBOL_TRADE_MODE_FULL and s.get('visible')
        ]

    def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """指定された金融商品の詳細情報を取得する"""
        info = mt5.symbol_info(symbol)
        if info is None:
            return None
        return info._asdict()

    def get_historical_data(self, symbol: str, timeframe_str: str, count: int, from_date: Optional[datetime] = None, to_date: Optional[datetime] = None) -> HistoricalData:
        """
        ヒストリカルデータを取得する。
        from_dateとto_dateが指定されていれば期間指定、なければcountでバーの数を指定する。
        """
        timeframe = self._TIMEFRAME_MAP.get(timeframe_str.upper())
        if timeframe is None:
            raise ValueError(f"Invalid timeframe string: {timeframe_str}")

        rates = None
        if from_date and to_date:
            utc_tz = pytz.utc
            from_date_utc = from_date.astimezone(utc_tz) if from_date.tzinfo else utc_tz.localize(from_date)
            to_date_utc = to_date.astimezone(utc_tz) if to_date.tzinfo else utc_tz.localize(to_date)
            rates = mt5.copy_rates_range(symbol, timeframe, from_date_utc, to_date_utc)
        else:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

        # --- 改善点: 戻り値の一貫性 ---
        if rates is None or len(rates) == 0:
            return []

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('utc').dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        return df.to_dict('records')

    def _get_filling_mode(self, symbol: str) -> int:
        """
        シンボルに最適な注文執行方針を決定する。
        ブローカーがサポートする執行方針を優先的に使用する。
        
        重要: filling_modeフィールドの値は、MT5の定数値と異なる場合がある。
        デバッグ結果によると:
        - filling_mode = 1 → FOK only
        - filling_mode = 2 → IOC only  
        - filling_mode = 4 → RETURN only
        - ビットマスクの組み合わせも可能
        """
        symbol_info = self.get_symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found.")

        filling_modes = symbol_info['filling_mode']
        
        logging.info(f"Symbol {symbol} filling_mode value: {filling_modes} (binary: {bin(filling_modes)})")
        
        # 実際のMT5定数値を確認してマッピング
        # ORDER_FILLING_FOK = 0, ORDER_FILLING_IOC = 1, ORDER_FILLING_RETURN = 2
        
        # ビットマスクとして解釈
        # bit 0 (値1) = FOK サポート
        # bit 1 (値2) = IOC サポート  
        # bit 2 (値4) = RETURN サポート
        
        # IOC (Immediate or Cancel)が許可されていれば、それを最優先で使用
        if filling_modes & 2:  # IOC bit
            logging.info(f"Filling mode for {symbol}: IOC is supported (bit 1 set), selecting IOC (value 1).")
            return 1  # MT5 constant for IOC
        
        # FOK (Fill or Kill)が許可されていれば、それを使用
        if filling_modes & 1:  # FOK bit
            logging.info(f"Filling mode for {symbol}: FOK is supported (bit 0 set), selecting FOK (value 0).")
            return 0  # MT5 constant for FOK

        # RETURN (Standard market execution)が許可されていれば、それを使用
        if filling_modes & 4:  # RETURN bit
            logging.info(f"Filling mode for {symbol}: RETURN is supported (bit 2 set), selecting RETURN (value 2).")
            return 2  # MT5 constant for RETURN
        
        # 特殊ケース: filling_mode値が直接MT5定数を示している可能性
        if filling_modes in [0, 1, 2]:
            logging.warning(f"Filling mode for {symbol}: Using raw value {filling_modes} as filling mode.")
            return filling_modes
        
        # フォールバック
        logging.error(f"Filling mode for {symbol}: No supported filling mode found! Available modes: {filling_modes}")
        return 0  # FOK as fallback

    def place_order(self, request: Dict[str, Any]) -> TradeResult:
        """取引リクエストを送信する。送信前にorder_checkで事前検証を行う。"""
        symbol = request.get("symbol")
        if not symbol:
            raise ValueError("Symbol must be specified in the request.")

        if "comment" in request:
            if not isinstance(request["comment"], str):
                request["comment"] = str(request["comment"])
            if len(request["comment"]) > 31:
                logging.warning(f"Comment is too long ({len(request['comment'])} chars). Truncating to 31. Original: {request['comment']}")
                request["comment"] = request["comment"][:31]

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found or not provided by the broker.")
            
        if not symbol_info.visible:
            logging.info(f"Symbol {symbol} is not visible in Market Watch, attempting to select it.")
            if not mt5.symbol_select(symbol, True):
                logging.error(f"Failed to select symbol {symbol}")
                raise TradeExecutionError(f"Failed to select symbol {symbol}. Please add it to Market Watch manually.")
            
            # Wait longer for the selection to take effect
            import time
            time.sleep(0.5)
            
            # Verify the symbol is now visible
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or not symbol_info.visible:
                logging.error(f"Symbol {symbol} still not visible after selection attempt")
                raise TradeExecutionError(f"Cannot make {symbol} visible. Please manually add it to Market Watch in MT5.")

        # 執行方針がリクエストになければ、最適なものを自動で取得・設定
        if "type_filling" not in request:
            request["type_filling"] = self._get_filling_mode(symbol)
        
        if "type_time" not in request:
            request["type_time"] = mt5.ORDER_TIME_GTC

        logging.debug(f"Checking order: {request}")
        check_result = mt5.order_check(request)

        if check_result is None:
            error_code, error_message_text = mt5.last_error()
            full_error_message = f"Order check failed to execute (returned None). Ensure symbol '{symbol}' is available and tradable. MT5 last_error: ({error_code}, '{error_message_text}')"
            logging.error(full_error_message)
            raise TradeExecutionError(full_error_message)

        # Important: retcode 0 means "Done" (success) for order_check
        # Only retcode 10009 (TRADE_RETCODE_DONE) means success for order_send
        if check_result.retcode != 0:  # 0 = success for order_check
            error_comment = check_result.comment
            retcode = check_result.retcode
            full_error_message = f"Order check failed: {error_comment} (retcode: {retcode})"
            logging.error(full_error_message)
            raise TradeExecutionError(full_error_message)
        
        logging.info("Order check successful. Sending order...")
        logging.debug(f"Sending order: {request}")
        
        result = mt5.order_send(request)
        # For order_send, success is indicated by retcode 10009 (TRADE_RETCODE_DONE)
        if result.retcode != 10009:  # 10009 = TRADE_RETCODE_DONE
            error_comment = result.comment
            retcode = result.retcode
            full_error_message = f"Order send failed: {error_comment} (retcode: {retcode})"
            logging.error(full_error_message)
            raise TradeExecutionError(full_error_message)

        logging.info(f"Order sent successfully. Deal: {result.deal}, Order: {result.order}")
        return result._asdict()

    def get_open_positions(self, symbol: Optional[str] = None, ticket: Optional[int] = None) -> List[PositionInfo]:
        """オープンなポジションを取得する"""
        if ticket:
            positions = mt5.positions_get(ticket=ticket)
        elif symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()
        
        if positions is None:
            return []
        
        return [p._asdict() for p in positions]

    def get_pending_orders(self, symbol: Optional[str] = None, ticket: Optional[int] = None) -> List[OrderInfo]:
        """待機中の注文（ペンディングオーダー）を取得する"""
        if ticket:
            orders = mt5.orders_get(ticket=ticket)
        elif symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()
            
        if orders is None:
            return []
            
        return [o._asdict() for o in orders]

    def close_position(self, ticket: int, volume: Optional[float] = None, deviation: int = 20) -> TradeResult:
        """指定されたチケットのポジションを決済する。volumeが指定されない場合は全量を決済する。"""
        # --- バグ修正: positionsはリストで返るため、最初の要素を取得 ---
        positions = self.get_open_positions(ticket=ticket)
        if not positions:
            raise ValueError(f"Position with ticket {ticket} not found.")
        
        position = positions[0]
        symbol = position['symbol']
        close_volume = volume if volume is not None else position['volume']

        if close_volume > position['volume']:
            raise ValueError(f"Close volume {close_volume} is greater than position volume {position['volume']}.")

        price = mt5.symbol_info_tick(symbol).bid if position['type'] == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(close_volume),
            "type": mt5.ORDER_TYPE_SELL if position['type'] == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": position['magic'],
            "comment": "Closed by API",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._get_filling_mode(symbol),
        }
        
        return self.place_order(request)

    def modify_position_sltp(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> TradeResult:
        """オープンなポジションのStop LossまたはTake Profitを変更する"""
        # --- バグ修正: positionsはリストで返るため、最初の要素を取得 ---
        positions = self.get_open_positions(ticket=ticket)
        if not positions:
            raise ValueError(f"Position with ticket {ticket} not found.")
        position = positions[0]

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": position['symbol'],
            "sl": sl if sl is not None else 0.0,
            "tp": tp if tp is not None else 0.0,
        }
        
        result = mt5.order_send(request)
        if result.retcode != 10009:  # 10009 = TRADE_RETCODE_DONE
            logging.error(f"Position modify failed: {result.comment}")
            raise TradeExecutionError(f"Position modify failed: {result.comment} (retcode: {result.retcode})")
        
        logging.info(f"Position {ticket} modified successfully.")
        return result._asdict()

    def cancel_pending_order(self, ticket: int) -> TradeResult:
        """待機中の注文（ペンディングオーダー）をキャンセルする"""
        # --- バグ修正: ordersはリストで返るため、存在チェックを行う ---
        # (ticket指定の場合は要素が1つのはずなので、リストが空でないことを確認すれば良い)
        orders = self.get_pending_orders(ticket=ticket)
        if not orders:
            raise ValueError(f"Pending order with ticket {ticket} not found.")

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": ticket,
        }
        
        result = mt5.order_send(request)
        if result.retcode != 10009:  # 10009 = TRADE_RETCODE_DONE
            logging.error(f"Order cancel failed: {result.comment}")
            raise TradeExecutionError(f"Order cancel failed: {result.comment} (retcode: {result.retcode})")

        logging.info(f"Pending order {ticket} canceled successfully.")
        return result._asdict()