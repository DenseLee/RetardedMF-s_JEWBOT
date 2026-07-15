"""Discord webhook logger for BTC bot trade actions."""
import requests, logging
from datetime import datetime

logger = logging.getLogger("BTCBot")

WEBHOOK_URL = "https://discordapp.com/api/webhooks/1506999636529905744/cFbLbpEu3lZI0VC0CPnYnwkvcL6hvMpWIGtoioVHJEwizaBbEBuDUzrpYIAcobT732Tq"

def _post(content):
    try:
        r = requests.post(WEBHOOK_URL, json={"content": content}, timeout=5)
        r.raise_for_status()
    except Exception as e:
        logger.warning(f"Discord webhook failed: {e}")

def log_signal(bot_id, direction, regime, confidence, price, atr):
    """H1 signal fired."""
    dir_str = "LONG" if direction == 1 else "SHORT"
    _post(f"**{dir_str} SIGNAL** | {bot_id}\n"
          f"Regime: {regime} | Conf: {confidence:.2f} | Price: ${price:,.2f} | ATR: ${atr:,.0f}")

def log_gated(bot_id, direction, reason):
    """Entry blocked by gate."""
    dir_str = "LONG" if direction == 1 else "SHORT"
    _post(f"**GATED** {dir_str} | {bot_id}\nReason: {reason}")

def log_entry(bot_id, direction, price, sl, tp, lots):
    """Trade entered."""
    dir_str = "LONG" if direction == 1 else "SHORT"
    rr = (tp - price) / (price - sl) if direction == 1 else (price - tp) / (sl - price)
    _post(f"**ENTER {dir_str}** | {bot_id}\n"
          f"Price: ${price:,.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f} | RR: 1:{rr:.1f}\n"
          f"Lots: {lots:.4f} BTC")

def log_exit(bot_id, direction, entry_price, exit_price, pnl_r, pnl_dollar, reason, mfe):
    """Trade closed."""
    dir_str = "LONG" if direction == 1 else "SHORT"
    emoji = "🟢" if pnl_r > 0 else "🔴"
    _post(f"{emoji} **EXIT {dir_str}** | {bot_id} | {reason}\n"
          f"Entry: ${entry_price:,.2f} → Exit: ${exit_price:,.2f}\n"
          f"PnL: {pnl_r:+.2f}R (${pnl_dollar:+.2f}) | MFE: {mfe:+.2f}R")
