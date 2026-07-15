"""
Phased Trade Manager — wide SL survival + wrong-detection + trailing.

Phase 1 (bars 0-3): Wide SL at phase1_sl × ATR. No breakeven, no trailing.
Phase 2 (bar 4):     Wrong-detection check — close if MFE < phase2_mfe_min.
Phase 3 (bars 4-12): Trailing at phase3_trail × ATR from best price.
Phase 4 (bars 12-18): Tighten trail to phase4_trail × ATR.
Bar 18:              Time stop.

R is denominated in Phase 1 SL units (phase1_sl × ATR).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Phase(Enum):
    PHASE1_SURVIVAL = 0
    PHASE2_DETECT = 1
    PHASE3_TRAILING = 2
    PHASE4_PRESSURE = 3
    TIME_STOP = 4


class TradeActionType(Enum):
    HOLD = "hold"
    CLOSE = "close"
    MODIFY_SL = "modify_sl"


@dataclass
class TradeAction:
    action_type: TradeActionType = TradeActionType.HOLD
    close_pct: float = 0.0
    new_sl: Optional[float] = None
    reason: str = ""


@dataclass
class TradeState:
    direction: int = 0
    entry_price: float = 0.0
    entry_atr: float = 0.0
    initial_sl: float = 0.0     # Phase 1 SL price
    hard_tp: float = 0.0
    current_sl: float = 0.0
    current_tp: float = 0.0
    lots: float = 0.0
    best_price: float = 0.0
    bars_held: int = 0
    phase: Phase = Phase.PHASE1_SURVIVAL
    trailing_active: bool = False
    regime: str = ""
    unrealized_pnl_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0


class TradeManager:
    """Phased exit management.

    Phase 1: Wide SL gives trade room to develop. No premature breakeven.
    Phase 2: Wrong-detection at phase1_bars — kill stalled trades.
    Phase 3: Trail profits on trades that proved direction.
    Phase 4: Tighten trail as time runs out.
    """

    def __init__(self, phase1_sl=1.5, hard_tp=1.4,
                 phase1_bars=4, phase2_mfe_min=0.1,
                 phase3_trail=0.75, phase4_trail=0.40,
                 phase4_start=12, max_hold=18):
        self.phase1_sl = phase1_sl          # wide initial SL (ATR multiplier)
        self.hard_tp = hard_tp              # TP (ATR multiplier)
        self.phase1_bars = phase1_bars      # bars in Phase 1
        self.phase2_mfe_min = phase2_mfe_min  # min MFE at Phase 2 check to survive
        self.phase3_trail = phase3_trail    # trail distance Phase 3 (ATR)
        self.phase4_trail = phase4_trail    # trail distance Phase 4 (ATR)
        self.phase4_start = phase4_start    # when Phase 4 starts (bars)
        self.max_hold = max_hold
        self.state: Optional[TradeState] = None

    # -- Entry --
    def enter(self, direction, entry_price, atr, lots, regime=""):
        sl_dist = self.phase1_sl * atr
        tp_dist = self.hard_tp * atr
        if direction == 1:
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        self.state = TradeState(
            direction=direction, entry_price=entry_price, entry_atr=atr,
            initial_sl=sl, hard_tp=tp, current_sl=sl, current_tp=tp,
            lots=lots, best_price=entry_price, regime=regime,
            phase=Phase.PHASE1_SURVIVAL,
        )
        return self.state

    # -- Update (called every M15 bar) --
    def update(self, current_price, current_high, current_low, current_atr, regime=""):
        if self.state is None:
            return TradeAction()

        s = self.state
        s.bars_held += 1

        # Update best price and R-metrics (R uses Phase 1 SL distance)
        sl_r_dist = s.entry_atr * self.phase1_sl
        if s.direction == 1:
            s.best_price = max(s.best_price, current_high)
            profit_r = (current_price - s.entry_price) / sl_r_dist
            mfe_r = (s.best_price - s.entry_price) / sl_r_dist
        else:
            s.best_price = min(s.best_price, current_low)
            profit_r = (s.entry_price - current_price) / sl_r_dist
            mfe_r = (s.entry_price - s.best_price) / sl_r_dist

        s.unrealized_pnl_r = profit_r
        s.mfe_r = max(s.mfe_r, mfe_r)

        # -- Time Stop --
        if s.bars_held >= self.max_hold:
            return TradeAction(TradeActionType.CLOSE, close_pct=1.0, reason="Time stop")

        # -- Phase 1: Survival (wide SL, no breakeven, no trail) --
        if s.bars_held <= self.phase1_bars:
            s.phase = Phase.PHASE1_SURVIVAL
            return TradeAction()  # let SL/TP handle it in the backtester

        # -- Phase 2: Wrong-detection check (at phase1_bars + 1) --
        if s.bars_held == self.phase1_bars + 1:
            s.phase = Phase.PHASE2_DETECT
            if s.mfe_r < self.phase2_mfe_min:
                return TradeAction(TradeActionType.CLOSE, close_pct=1.0,
                                   reason=f"Wrong detection: MFE={s.mfe_r:.3f}R < {self.phase2_mfe_min}")

        # -- Phase 3: Trailing --
        if s.bars_held <= self.phase4_start:
            s.phase = Phase.PHASE3_TRAILING
            new_sl = self._compute_trail_sl(s, self.phase3_trail)
            if self._sl_improved(new_sl, s.current_sl):
                s.current_sl = new_sl
                return TradeAction(TradeActionType.MODIFY_SL, new_sl=new_sl,
                                   reason=f"Trail to {new_sl:.2f}")
            return TradeAction()

        # -- Phase 4: Time pressure --
        if s.bars_held <= self.max_hold:
            s.phase = Phase.PHASE4_PRESSURE
            new_sl = self._compute_trail_sl(s, self.phase4_trail)
            if self._sl_improved(new_sl, s.current_sl):
                s.current_sl = new_sl
                return TradeAction(TradeActionType.MODIFY_SL, new_sl=new_sl,
                                   reason=f"Pressure trail to {new_sl:.2f}")
            return TradeAction()

        return TradeAction()

    # -- Trail SL computation --
    def _compute_trail_sl(self, s, trail_atr_mult):
        """Trail SL at trail_atr_mult × ATR from best price, never worse than entry."""
        d = trail_atr_mult * s.entry_atr
        if s.direction == 1:
            return max(s.best_price - d, s.entry_price)
        return min(s.best_price + d, s.entry_price)

    def _sl_improved(self, new_sl, cur):
        if self.state is None:
            return False
        return (new_sl > cur + 0.01) if self.state.direction == 1 else (new_sl < cur - 0.01)

    # -- Helpers --
    def check_sl_hit(self, low, high):
        if self.state is None:
            return False
        return low <= self.state.current_sl if self.state.direction == 1 else high >= self.state.current_sl

    def check_tp_hit(self, low, high):
        if self.state is None:
            return False
        return high >= self.state.current_tp if self.state.direction == 1 else low <= self.state.current_tp

    def exit_price_at_sl(self):
        return self.state.current_sl if self.state else 0.0

    def exit_price_at_tp(self):
        return self.state.current_tp if self.state else 0.0

    def get_summary(self):
        if self.state is None:
            return {"active": False}
        s = self.state
        return {
            "active": True, "direction": "LONG" if s.direction == 1 else "SHORT",
            "entry_price": s.entry_price, "current_sl": s.current_sl,
            "current_tp": s.current_tp, "lots": s.lots, "bars_held": s.bars_held,
            "phase": s.phase.name, "unrealized_pnl_r": round(s.unrealized_pnl_r, 4),
            "mfe_r": round(s.mfe_r, 4),
        }

    @staticmethod
    def compute_position_size(capital, atr, entry_price, risk_pct=0.02, initial_sl=1.5,
                              min_lot=0.01, lot_step=0.01, max_lot=0.5):
        """Dollar-risk position sizing. SL is in ATR units."""
        dollar_risk = capital * risk_pct
        sl_distance = initial_sl * atr
        if sl_distance < 1e-9:
            return min_lot
        position_size = dollar_risk / sl_distance
        position_size = min(position_size, max_lot)
        position_size = max(min_lot, round(position_size / lot_step) * lot_step)
        return position_size
