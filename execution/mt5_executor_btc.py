"""
MT5 Executor for BTC/USDT — order placement, position management, account info.
Adapted from mt5_inference_engine.py lines 302-940.

Handles:
  - MT5 connection with retry logic
  - Order placement (market orders with SL/TP)
  - Position closing and modification
  - Lot size computation for BTC (contract_size=1, min=0.01, step=0.001)
  - Account balance and position queries
"""
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# MT5 constants (will be populated after import)
ORDER_FILLING_IOC = 1
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
TRADE_ACTION_DEAL = 1
TRADE_ACTION_SLTP = 6
TRADE_RETCODE_DONE = 10009
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1


@dataclass
class MT5Position:
    ticket: int
    symbol: str
    type: int       # 0=Buy, 1=Sell
    volume: float
    price_open: float
    sl: float
    tp: float
    profit: float
    comment: str


@dataclass
class MT5OrderResult:
    success: bool
    ticket: Optional[int] = None
    price: float = 0.0
    volume: float = 0.0
    error: str = ""


class MT5Executor:
    """MT5 connection and order execution."""

    def __init__(self, symbol="BTCUSD", magic=20260517, deviation=20):
        self.symbol = symbol
        self.magic = magic
        self.deviation = deviation
        self.mt5 = None
        self.connected = False
        self.contract_size = 1.0
        self.volume_min = 0.01
        self.volume_max = 100.0
        self.volume_step = 0.001
        self.point_size = 1.0

    # ═══════════════════════════════════════════════
    # Connection
    # ═══════════════════════════════════════════════

    def connect(self) -> bool:
        """Connect to MT5. Returns True on success."""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5
        except ImportError:
            logger.error("MetaTrader5 module not installed")
            return False

        if not self.mt5.initialize():
            logger.error(f"MT5 initialize() failed: {self.mt5.last_error()}")
            return False

        self.connected = True
        logger.info(f"MT5 connected, terminal: {self.mt5.terminal_info()}")

        # Get symbol info
        self._update_symbol_info()
        return True

    def is_connected(self):
        return self.connected

    def disconnect(self):
        """Disconnect from MT5."""
        if self.mt5 and self.connected:
            self.mt5.shutdown()
            self.connected = False
            logger.info("MT5 disconnected")

    def health_check(self) -> bool:
        """Check if MT5 is still healthy."""
        if not self.connected or not self.mt5:
            return False
        try:
            info = self.mt5.terminal_info()
            return info is not None
        except Exception:
            return False

    def reconnect(self, max_attempts=3) -> bool:
        """Attempt reconnection with retries."""
        for attempt in range(max_attempts):
            logger.info(f"Reconnect attempt {attempt + 1}/{max_attempts}")
            if self.connect():
                return True
            time.sleep(5)
        return False

    # ═══════════════════════════════════════════════
    # Symbol info
    # ═══════════════════════════════════════════════

    def _update_symbol_info(self):
        """Update symbol trading parameters."""
        if not self.connected:
            return
        try:
            info = self.mt5.symbol_info(self.symbol)
            if info is not None:
                self.contract_size = info.trade_contract_size or 1.0
                self.volume_min = info.volume_min or 0.01
                self.volume_max = info.volume_max or 100.0
                self.volume_step = info.volume_step or 0.001
                self.point_size = info.point or 1.0
                logger.info(f"Symbol {self.symbol}: contract={self.contract_size}, "
                           f"min_lot={self.volume_min}, step={self.volume_step}")
            else:
                logger.warning(f"Symbol {self.symbol} not found in MT5")
        except Exception as e:
            logger.error(f"Symbol info error: {e}")

    def get_current_price(self) -> Optional[float]:
        """Get current bid/ask mid price."""
        if not self.connected:
            return None
        try:
            tick = self.mt5.symbol_info_tick(self.symbol)
            if tick:
                return (tick.bid + tick.ask) / 2.0
        except Exception:
            pass
        return None

    # ═══════════════════════════════════════════════
    # Lot size computation
    # ═══════════════════════════════════════════════

    def compute_lot_size(self, balance: float, risk_pct: float,
                         atr: float, sl_atr_mult: float = 1.0) -> float:
        """Compute lot size from risk parameters."""
        risk_amount = balance * risk_pct
        sl_points = atr * sl_atr_mult
        if sl_points < 1e-9 or self.contract_size < 1e-9:
            return self.volume_min

        raw_lots = risk_amount / (sl_points * self.contract_size)

        # Round to volume step
        lots = round(raw_lots / self.volume_step) * self.volume_step
        lots = max(self.volume_min, min(self.volume_max, lots))
        return lots

    # ═══════════════════════════════════════════════
    # Order execution
    # ═══════════════════════════════════════════════

    def open_position(self, direction: int, lots: float, sl: float, tp: float,
                      comment: str = "BTC_BOT") -> MT5OrderResult:
        """
        Open a new position.

        Args:
            direction: 1 = Buy, -1 = Sell
            lots: position size in BTC
            sl: stop loss price
            tp: take profit price
            comment: order comment

        Returns:
            MT5OrderResult with success, ticket, price
        """
        if not self.connected:
            return MT5OrderResult(False, error="Not connected to MT5")

        price = self.get_current_price()
        if price is None:
            return MT5OrderResult(False, error="Cannot get current price")

        # Build request
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_FOK,
        }

        try:
            result = self.mt5.order_send(request)
        except Exception as e:
            return MT5OrderResult(False, error=str(e))

        if result is None:
            return MT5OrderResult(False, error="order_send returned None")

        if result.retcode == self.mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order filled: ticket={result.order}, "
                       f"price={result.price}, vol={result.volume}")
            return MT5OrderResult(
                success=True, ticket=result.order,
                price=result.price, volume=result.volume)
        else:
            error_msg = f"retcode={result.retcode}, {result.comment}"
            logger.error(f"Order failed: {error_msg}")
            return MT5OrderResult(False, error=error_msg)

    def close_position(self, ticket: int, lots: float = None,
                       comment: str = "BTC_BOT_CLOSE") -> MT5OrderResult:
        """
        Close an existing position by ticket.
        If lots is None, closes the full position.
        """
        if not self.connected:
            return MT5OrderResult(False, error="Not connected to MT5")

        # Get position info
        positions = self.get_positions()
        pos = next((p for p in positions if p.ticket == ticket), None)
        if pos is None:
            return MT5OrderResult(False, error=f"Position {ticket} not found")

        close_lots = lots if lots is not None else pos.volume
        close_type = (self.mt5.ORDER_TYPE_SELL if pos.type == self.mt5.POSITION_TYPE_BUY
                     else self.mt5.ORDER_TYPE_BUY)

        price = self.get_current_price()
        if price is None:
            return MT5OrderResult(False, error="Cannot get current price")

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": close_lots,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic,
            "comment": comment,
            "type_filling": self.mt5.ORDER_FILLING_FOK,
        }

        try:
            result = self.mt5.order_send(request)
        except Exception as e:
            return MT5OrderResult(False, error=str(e))

        if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed at {result.price}")
            return MT5OrderResult(success=True, ticket=ticket, price=result.price,
                                 volume=close_lots)
        else:
            error = f"retcode={result.retcode if result else 'None'}"
            return MT5OrderResult(False, error=error)

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify SL and TP of an existing position."""
        if not self.connected:
            return False

        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }

        try:
            result = self.mt5.order_send(request)
        except Exception:
            return False

        if result and result.retcode == self.mt5.TRADE_RETCODE_DONE:
            logger.info(f"SL/TP modified: ticket={ticket}, sl={sl:.2f}, tp={tp:.2f}")
            return True
        return False

    # ═══════════════════════════════════════════════
    # Queries
    # ═══════════════════════════════════════════════

    def get_positions(self, symbol: str = None) -> list:
        """Get open positions, filtered by symbol if specified."""
        if not self.connected:
            return []
        sym = symbol or self.symbol
        try:
            raw = self.mt5.positions_get(symbol=sym)
            if raw is None:
                return []
            return [
                MT5Position(
                    ticket=p.ticket,
                    symbol=p.symbol,
                    type=p.type,
                    volume=p.volume,
                    price_open=p.price_open,
                    sl=p.sl,
                    tp=p.tp,
                    profit=p.profit,
                    comment=p.comment,
                )
                for p in raw
            ]
        except Exception:
            return []

    def get_account_balance(self) -> Optional[float]:
        """Get current account balance."""
        if not self.connected:
            return None
        try:
            info = self.mt5.account_info()
            return info.balance if info else None
        except Exception:
            return None

    def get_account_equity(self) -> Optional[float]:
        """Get current account equity (balance + floating PnL)."""
        if not self.connected:
            return None
        try:
            info = self.mt5.account_info()
            return info.equity if info else None
        except Exception:
            return None

    def get_bars(self, timeframe: str, count: int = 500) -> Optional[list]:
        """
        Fetch OHLCV bars from MT5.

        Args:
            timeframe: 'H1', 'M15', 'H4' etc.
            count: number of bars

        Returns:
            list of [timestamp, open, high, low, close, tick_volume, spread, real_volume]
        """
        if not self.connected:
            return None

        tf_map = {
            "M15": self.mt5.TIMEFRAME_M15, "15m": self.mt5.TIMEFRAME_M15,
            "H1": self.mt5.TIMEFRAME_H1, "1h": self.mt5.TIMEFRAME_H1,
            "H4": self.mt5.TIMEFRAME_H4, "4h": self.mt5.TIMEFRAME_H4,
        }

        mt5_tf = tf_map.get(timeframe)
        if mt5_tf is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        import concurrent.futures
        try:
            # Run MT5 call in a thread with timeout to prevent hard hangs
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self.mt5.copy_rates_from_pos, self.symbol, mt5_tf, 0, count)
                rates = future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            logger.error(f"MT5 get_bars({timeframe}) timed out after 60s — "
                         f"terminal may be unresponsive, reconnecting...")
            try:
                self.mt5.shutdown()
            except Exception:
                pass
            time.sleep(5)
            if not self.connect():
                logger.error("MT5 reconnect failed")
            return None
        except Exception as e:
            logger.error(f"MT5 get_bars error: {e}")
            return None

        if rates is None or len(rates) == 0:
            return None
        import pandas as pd
        df = pd.DataFrame(rates)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df


# ═══════════════════════════════════════════════
# Dry-run mock for testing without MT5
# ═══════════════════════════════════════════════

class DryRunExecutor:
    """Mock executor for testing without MT5 connection."""

    def __init__(self, symbol="BTCUSD", initial_balance=10000.0):
        self.symbol = symbol
        self.connected = True
        self.contract_size = 1.0
        self.volume_min = 0.01
        self.volume_max = 100.0
        self.volume_step = 0.001
        self._balance = initial_balance
        self._positions = []
        self._next_ticket = 1000
        self._current_price = 80000.0
        self.mt5 = None

    def is_connected(self):
        return self.connected

    def connect(self): return True
    def disconnect(self): self.connected = False
    def health_check(self): return True
    def reconnect(self, max_attempts=3): return True

    def get_current_price(self): return self._current_price
    def set_price(self, price): self._current_price = price

    def compute_lot_size(self, balance, risk_pct, atr, sl_atr_mult=1.0):
        risk_amount = balance * risk_pct
        sl_points = atr * sl_atr_mult
        if sl_points < 1e-9:
            return self.volume_min
        raw_lots = risk_amount / (sl_points * self.contract_size)
        lots = round(raw_lots / self.volume_step) * self.volume_step
        return max(self.volume_min, min(self.volume_max, lots))

    def open_position(self, direction, lots, sl, tp, comment="BTC_BOT"):
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions.append(MT5Position(
            ticket=ticket, symbol=self.symbol,
            type=0 if direction == 1 else 1,
            volume=lots, price_open=self._current_price,
            sl=sl, tp=tp, profit=0.0, comment=comment))
        return MT5OrderResult(True, ticket=ticket, price=self._current_price,
                             volume=lots)

    def close_position(self, ticket, lots=None, comment="BTC_BOT_CLOSE"):
        pos = next((p for p in self._positions if p.ticket == ticket), None)
        if pos is None:
            return MT5OrderResult(False, error="Position not found")
        close_lots = lots or pos.volume
        if lots is not None and lots < pos.volume:
            pos.volume -= lots
        else:
            self._positions.remove(pos)
        # Compute realized PnL
        if pos.type == 0:  # Buy
            pnl = (self._current_price - pos.price_open) * close_lots * self.contract_size
        else:
            pnl = (pos.price_open - self._current_price) * close_lots * self.contract_size
        self._balance += pnl
        return MT5OrderResult(True, ticket=ticket, price=self._current_price,
                             volume=close_lots)

    def modify_sl_tp(self, ticket, sl, tp):
        pos = next((p for p in self._positions if p.ticket == ticket), None)
        if pos:
            pos.sl = sl
            pos.tp = tp
            return True
        return False

    def get_positions(self, symbol=None):
        return self._positions

    def get_account_balance(self):
        return self._balance

    def get_account_equity(self):
        equity = self._balance
        for p in self._positions:
            if p.type == 0:  # Buy
                equity += (self._current_price - p.price_open) * p.volume * self.contract_size
            else:
                equity += (p.price_open - self._current_price) * p.volume * self.contract_size
        return equity
