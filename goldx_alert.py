#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import math
import mimetypes
from dotenv import load_dotenv

import smtplib
import ssl
import sys
import time
from twelvedata import TDClient
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone, tzinfo
from email.message import EmailMessage
from typing import Any, Iterable, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates



from zoneinfo import ZoneInfo



@dataclass(frozen=True)
class PricePoint:
    ts: datetime
    price: float


_TD_CLIENTS: dict[str, TDClient] = {}


def _get_td_client(api_key: str) -> TDClient:
    client = _TD_CLIENTS.get(api_key)
    if client is None:
        client = TDClient(apikey=api_key)
        _TD_CLIENTS[api_key] = client
    return client


def _local_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo or timezone.utc


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _parse_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


def _parse_api_keys(raw: str | None) -> list[str]:
    if raw is None:
        return []
    # Allow comma / whitespace separated tokens.
    tokens: list[str] = []
    for part in raw.split(","):
        t = part.strip()
        if not t:
            continue
        tokens.append(t)
    return tokens


def _get_arg_value(argv: list[str], key: str, default: Optional[str] = None) -> Optional[str]:
    for i, arg in enumerate(argv):
        if arg == key and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(key + "="):
            return arg.split("=", 1)[1]
    return default


def _parse_date(value: str) -> date:
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def _today_in_tz(tz: tzinfo) -> date:
    return datetime.now(tz).date()


def _sanitize_symbol_for_path(symbol: str) -> str:
    out = []
    for ch in symbol.strip():
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "symbol"




def _parse_dt_raw(value: str) -> datetime:
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")


class DataProviderError(RuntimeError):
    pass


def _format_utc_offset_name(offset: timedelta) -> str:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    if minutes:
        return f"UTC{sign}{hours:02d}:{minutes:02d}"
    return f"UTC{sign}{hours}"



def fetch_twelvedata_1min_series(
    *,
    api_key: str,
    symbol: str,
    interval: str = "1min",
    outputsize: int = 200,
    tz: tzinfo ,
    timezone_name: str = "Asia/Shanghai",
) -> list[PricePoint]:
    td = _get_td_client(api_key)
    ts=td.time_series(
        symbol=symbol,
        interval=interval,
        timezone=timezone_name,
        outputsize=outputsize,
        order="asc",
    )
    data=ts.as_json()
    points = []
    for row in data:
        dt_raw = row.get("datetime")
        close_raw = row.get("close")
        if dt_raw is None or close_raw is None:
            continue
        try:
            date = datetime.strptime(dt_raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz)
            price = float(close_raw)
        except Exception:
            continue
        points.append(PricePoint(ts=date, price=price))

    return points


def ema(series: Iterable[float], period: int) -> list[float]:
    series_list = list(series)
    if not series_list:
        return []
    alpha = 2.0 / (period + 1)
    out: list[float] = []
    value = series_list[0]
    out.append(value)
    for x in series_list[1:]:
        value = alpha * x + (1 - alpha) * value
        out.append(value)
    return out


def macd_histogram(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> list[float]:
    if len(prices) < slow + signal:
        return []
    ema_fast = ema(prices, fast)
    ema_slow = ema(prices, slow)
    macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
    signal_line = ema(macd_line, signal)
    return [m - s for m, s in zip(macd_line, signal_line)]


def linear_slope(prices: list[float]) -> float:
    n = len(prices)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(prices) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, prices))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den  # price change per bar (assumes 1 bar = 1 minute)

def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values is empty")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_vals = sorted(values)
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass(frozen=True)
class Signal:
    should_alert: bool
    reason: str
    metrics: dict[str, float]


def compute_buy_signal(
    points: list[PricePoint],
    *,
    low_window_min: int,
    slope_window_min: int,
    quantile_window_min: int,
    q_low: float,
    q_skip: float,
    stall_min: int,
    micro_rebound_pct: float,
    rebound_pct: float,
    rebound_max_pct: float,
    low_max_age_min: int,
    slope_threshold_pct_per_min: float,
    hist_rise_bars: int,
) -> Signal:
    if len(points) < max(40, low_window_min, slope_window_min):
        return Signal(False, "数据不足", {})

    if rebound_max_pct > 0 and rebound_max_pct < rebound_pct:
        rebound_max_pct = rebound_pct
    if q_low < 0:
        q_low = 0.0
    if q_low > 100:
        q_low = 100.0
    if q_skip < 0:
        q_skip = 0.0
    if q_skip > 100:
        q_skip = 100.0
    if q_skip > 0 and q_low > 0 and q_skip < q_low:
        q_skip = q_low
    q_low_eff = q_low if q_low > 0 else 25.0
    q_skip_eff = q_skip if q_skip > 0 else 60.0

    closes = [p.price for p in points]

    low_window_points = points[-low_window_min:]
    low_point = min(low_window_points, key=lambda p: p.price)
    low = low_point.price
    last = closes[-1]
    rebound = (last - low) / low * 100.0
    low_age_min = (points[-1].ts - low_point.ts).total_seconds() / 60.0

    if quantile_window_min <= 0:
        quantile_window = closes
    else:
        quantile_window = closes[-quantile_window_min:]
    if not quantile_window:
        quantile_window = [last]
    q_low_value = _quantile(quantile_window, q_low_eff)
    q_skip_value = _quantile(quantile_window, q_skip_eff)

    slope_window = closes[-slope_window_min:]
    slope_abs = linear_slope(slope_window)
    slope_pct_per_min = (slope_abs / last) * 100.0 if last != 0 else 0.0

    hist = macd_histogram(closes)
    hist_rising = False
    hist_last = []
    if hist and len(hist) >= hist_rise_bars:
        hist_last = hist[-hist_rise_bars:]
        hist_rising = all(a < b for a, b in zip(hist_last, hist_last[1:]))

    rebound_max_ok = rebound_max_pct <= 0 or rebound <= rebound_max_pct
    rebound_ok = rebound >= rebound_pct and rebound_max_ok
    low_age_ok = low_max_age_min <= 0 or low_age_min <= low_max_age_min
    micro_rebound_ok = micro_rebound_pct <= 0 or rebound >= micro_rebound_pct
    stall_ok = True
    if stall_min > 0:
        stall_window = points[-stall_min:] if stall_min <= len(points) else points
        stall_low = min(p.price for p in stall_window) if stall_window else last
        stall_ok = (points[-1].ts - low_point.ts).total_seconds() / 60.0 >= stall_min and stall_low >= low_point.price

    if q_skip > 0 and last > q_skip_value:
        metrics = {
            "last": last,
            "low": low,
            "rebound_pct": rebound,
            "slope_pct_per_min": slope_pct_per_min,
            "low_age_min": low_age_min,
            "q_low_value": q_low_value,
            "q_skip_value": q_skip_value,
            "quantile_window_min": float(quantile_window_min),
        }
        if hist_last:
            metrics["macd_hist_last"] = hist_last[-1]
        return Signal(False, f"价格高于{q_skip:.1f}分位过滤", metrics)

    low_zone = q_low > 0 and last <= q_low_value
    if low_zone:
        should = low_age_ok and stall_ok and micro_rebound_ok and rebound_max_ok and slope_pct_per_min >= slope_threshold_pct_per_min
        zone_label = "低分位提前"
    else:
        should = rebound_ok and low_age_ok and slope_pct_per_min >= slope_threshold_pct_per_min and hist_rising
        zone_label = "中分位确认"

    metrics = {
        "last": last,
        "low": low,
        "rebound_pct": rebound,
        "slope_pct_per_min": slope_pct_per_min,
        "low_age_min": low_age_min,
        "q_low_value": q_low_value,
        "q_skip_value": q_skip_value,
        "quantile_window_min": float(quantile_window_min),
    }
    if hist_last:
        metrics["macd_hist_last"] = hist_last[-1]
    rebound_part = f"反弹{rebound:.3f}%>=阈值{rebound_pct:.3f}%"
    if rebound_max_pct > 0:
        rebound_part += f"且<=上限{rebound_max_pct:.3f}%"
    low_age_part = ""
    if low_max_age_min > 0:
        low_age_part = f"低点距今{low_age_min:.1f}min<=阈值{low_max_age_min}"
    q_part = f"分位q{q_low_eff:.0f}/q{q_skip_eff:.0f}={q_low_value:.4f}/{q_skip_value:.4f}"
    if low_zone:
        early_part = f"微反弹>={micro_rebound_pct:.3f}%={micro_rebound_ok}"
        if rebound_max_pct > 0:
            early_part += f"，反弹<=上限{rebound_max_pct:.3f}%={rebound_max_ok}"
        age_part = f"，{low_age_part}" if low_age_part else ""
        reason = (
            f"{zone_label}：{q_part}{age_part}，停跌{stall_ok}，{early_part}，"
            f"斜率{slope_pct_per_min:.4f}%/min>=阈值{slope_threshold_pct_per_min:.4f}"
        )
    else:
        age_part = f"，{low_age_part}" if low_age_part else ""
        reason = (
            f"{zone_label}：{q_part}，{rebound_part}{age_part}，"
            f"斜率{slope_pct_per_min:.4f}%/min>=阈值{slope_threshold_pct_per_min:.4f}，"
            f"MACD柱体连续回升{hist_rise_bars}根={hist_rising}"
        )
    return Signal(should, reason, metrics)


@dataclass(frozen=True)
class BacktestAlert:
    ts: datetime
    price: float
    low_window_ts: datetime
    low_window_price: float
    low_so_far_ts: datetime
    low_so_far_price: float
    reason: str
    metrics: dict[str, float]


def _min_price_point(points: list[PricePoint]) -> PricePoint:
    if not points:
        raise ValueError("points is empty")
    return min(points, key=lambda p: p.price)


def _write_csv(path: str, header: list[str], rows: list[list[Any]]) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _load_points_cache(path: str, *, tz: tzinfo) -> Optional[list[PricePoint]]:
    """从缓存加载数据点"""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            items = raw.get("points", [])
        else:
            return None

        points: list[PricePoint] = []
        for row in items:
            if not isinstance(row, dict):
                continue
            raw_ts = row.get("ts")
            price = row.get("price")
            if raw_ts is None or price is None:
                continue
            points.append(PricePoint(ts=datetime.fromisoformat(raw_ts).replace(tzinfo=tz), price=float(price)))
        if not points:
            return None
        return points
    except Exception as e:
        print(f"缓存加载失败：{e}",file=sys.stderr)
        return None


def _save_points_cache(path: str, points: list[PricePoint],tz="Asia/Shanghai") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    raw = {
        "saved_at": datetime.now(ZoneInfo(tz)).isoformat(),
        "points": [{"ts": p.ts.isoformat(), "price": p.price} for p in points],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False)





def _plot_full_day(
    *,
    points: list[PricePoint],
    alerts: list[BacktestAlert],
    out_path: str,
    title: str,
    tz: tzinfo,
) -> bool:


    if not points:
        return False

    xs = [p.ts for p in points]
    ys = [p.price for p in points]
    low_p = _min_price_point(points)

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.scatter([low_p.ts], [low_p.price], color="#2ca02c", s=40, zorder=5, label="Low")

    if alerts:
        plt.scatter([a.ts for a in alerts], [a.price for a in alerts], color="#d62728", s=30, zorder=6, label="Alert")

    ax = plt.gca()
    try:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(0, 24, 3)), tz=tz))
    except Exception:
        pass
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _plot_alert_snapshot(
    *,
    points: list[PricePoint],
    alert: BacktestAlert,
    out_path: str,
    title: str,
    tz: tzinfo,
) -> bool:

    if not points:
        return False

    xs = [p.ts for p in points]
    ys = [p.price for p in points]

    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, linewidth=1.2)
    plt.scatter([alert.low_so_far_ts], [alert.low_so_far_price], color="#2ca02c", s=45, zorder=6, label="Low so far")
    plt.scatter([alert.low_window_ts], [alert.low_window_price], color="#1f77b4", s=40, zorder=6, label="Low in window")
    plt.scatter([alert.ts], [alert.price], color="#d62728", s=55, zorder=7, label="Alert")

    ax = plt.gca()
    try:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=list(range(0, 24, 3)), tz=tz))
    except Exception:
        pass
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    info = (
        f"rebound={alert.metrics.get('rebound_pct', float('nan')):.3f}%\n"
        f"slope={alert.metrics.get('slope_pct_per_min', float('nan')):.4f}%/min\n"
        f"reason={alert.reason}"
    )
    plt.gcf().text(0.01, 0.01, info, fontsize=9, va="bottom", ha="left", wrap=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True


def _build_realtime_alert(points: list[PricePoint], cfg: RuntimeConfig, signal: Signal) -> BacktestAlert:
    if not points:
        raise ValueError("points is empty")
    last = points[-1]
    low_window_points = points[-cfg.low_window_min:] if cfg.low_window_min <= len(points) else points
    low_window = min(low_window_points, key=lambda p: p.price)
    low_so_far = min(points, key=lambda p: p.price)
    return BacktestAlert(
        ts=last.ts,
        price=last.price,
        low_window_ts=low_window.ts,
        low_window_price=low_window.price,
        low_so_far_ts=low_so_far.ts,
        low_so_far_price=low_so_far.price,
        reason=signal.reason,
        metrics=signal.metrics,
    )


def _render_realtime_alert_plot(points: list[PricePoint], cfg: RuntimeConfig, signal: Signal) -> Optional[str]:
    if not points:
        return None
    alert = _build_realtime_alert(points, cfg, signal)
    date_str = alert.ts.astimezone(cfg.monitor.tz).strftime("%Y-%m-%d")
    symbol_slug = _sanitize_symbol_for_path(cfg.symbol)
    out_dir = os.path.join(cfg.alert_plot_dir, symbol_slug, date_str)
    os.makedirs(out_dir, exist_ok=True)
    fn = f"alert_{alert.ts.astimezone(cfg.monitor.tz).strftime('%Y-%m-%d_%H%M')}.png"
    out_path = os.path.join(out_dir, fn)
    ok = _plot_alert_snapshot(
        points=points,
        alert=alert,
        out_path=out_path,
        title=f"{cfg.symbol} realtime alert {alert.ts.strftime('%Y-%m-%d %H:%M %Z')}",
        tz=cfg.monitor.tz,
    )
    return out_path if ok else None


@dataclass
class EmailConfig:
    host: str
    port: int
    username: str
    password: str
    from_addr: str
    to_addrs: list[str]
    use_ssl: bool = True
    use_starttls: bool = False


def send_email(cfg: EmailConfig, *, subject: str, body: str, attachments: Optional[list[str]] = None) -> None:
    msg = EmailMessage()
    msg["From"] = cfg.from_addr
    msg["To"] = ", ".join(cfg.to_addrs)
    msg["Subject"] = subject
    msg.set_content(body)
    if attachments:
        for path in attachments:
            if not path:
                continue
            if not os.path.exists(path):
                print(f"附件不存在，已跳过：{path}", file=sys.stderr)
                continue
            ctype, _ = mimetypes.guess_type(path)
            if not ctype:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            with open(path, "rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(path))

    if cfg.use_ssl:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(cfg.host, cfg.port, context=context, timeout=20) as s:
            s.login(cfg.username, cfg.password)
            s.send_message(msg)
        return

    with smtplib.SMTP(cfg.host, cfg.port, timeout=20) as s:
        s.ehlo()
        if cfg.use_starttls:
            context = ssl.create_default_context()
            s.starttls(context=context)
            s.ehlo()
        s.login(cfg.username, cfg.password)
        s.send_message(msg)


@dataclass
class RuntimeConfig:
    provider: str
    poll_seconds: int
    cooldown_minutes: int
    dry_run: bool
    symbol: str
    twelvedata_timezone: str
    state_file: str
    monitor: "MonitorWindow"
    alert_plot_on_email: bool
    alert_plot_dir: str
    # strategy
    low_window_min: int
    slope_window_min: int
    quantile_window_min: int
    q_low: float
    q_skip: float
    stall_min: int
    micro_rebound_pct: float
    rebound_pct: float
    rebound_max_pct: float
    low_max_age_min: int
    slope_threshold_pct_per_min: float
    hist_rise_bars: int


@dataclass(frozen=True)
class MonitorWindow:
    tz: tzinfo
    tz_name: str
    start_min: int  # minutes from midnight; 0..1440
    end_min: int  # minutes from midnight; 0..1440 (allow 1440 as 24:00)


def _parse_hhmm_to_minutes(value: str) -> int:
    v = value.strip()
    if v == "":
        raise ValueError("时间不能为空")
    if v == "24:00":
        return 1440
    parts = v.split(":")
    if len(parts) != 2:
        raise ValueError(f"时间格式应为 HH:MM（如 09:00），收到：{value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23):
        raise ValueError(f"小时应在 0..23，收到：{value!r}")
    if not (0 <= minute <= 59):
        raise ValueError(f"分钟应在 0..59，收到：{value!r}")
    return hour * 60 + minute


def _resolve_tz(name: str) -> tzinfo:
    raw = name.strip()
    if raw == "" or raw.lower() == "local":
        return _local_tz()

    if ZoneInfo is not None:
        try:
            return ZoneInfo(raw)
        except Exception:
            pass

    s = raw.upper()
    if s.startswith("UTC"):
        s = s[3:]
    if not s:
        return timezone.utc

    sign = 1
    if s[0] == "+":
        s = s[1:]
    elif s[0] == "-":
        sign = -1
        s = s[1:]

    if ":" in s:
        h_str, m_str = s.split(":", 1)
        hours = int(h_str)
        minutes = int(m_str)
    else:
        hours = int(s)
        minutes = 0
    if not (0 <= hours <= 23) or not (0 <= minutes <= 59):
        raise ValueError(f"无法解析 MONITOR_TZ：{name!r}")
    return timezone(sign * timedelta(hours=hours, minutes=minutes))


def _load_monitor_window() -> MonitorWindow:
    tz_name = os.environ.get("MONITOR_TZ", "").strip() or "local"
    start_raw = os.environ.get("MONITOR_START", "").strip()
    end_raw = os.environ.get("MONITOR_END", "").strip()

    if start_raw == "" and end_raw == "":
        start_min = 0
        end_min = 1440
    else:
        start_min = _parse_hhmm_to_minutes(start_raw) if start_raw else 0
        end_min = _parse_hhmm_to_minutes(end_raw) if end_raw else 1440

    tz = _resolve_tz(tz_name)
    return MonitorWindow(tz=tz, tz_name=tz_name, start_min=start_min, end_min=end_min)


def _is_within_window(now: datetime, window: MonitorWindow) -> bool:
    start_min = window.start_min
    end_min = window.end_min
    if start_min == 0 and end_min == 1440:
        return True

    current = now.hour * 60 + now.minute + (now.second / 60.0)
    if start_min < end_min:
        return start_min <= current < end_min
    return current >= start_min or current < end_min


def _dt_at_minutes(now: datetime, *, minutes: int) -> datetime:
    base = datetime(now.year, now.month, now.day, 0, 0, 0, tzinfo=now.tzinfo)
    if minutes == 1440:
        return base + timedelta(days=1)
    return base + timedelta(minutes=minutes)


def _format_minutes(minutes: int) -> str:
    if minutes == 1440:
        return "24:00"
    return f"{minutes//60:02d}:{minutes%60:02d}"


def _next_window_start(now: datetime, window: MonitorWindow) -> datetime:
    start_min = window.start_min
    end_min = window.end_min
    current = now.hour * 60 + now.minute + (now.second / 60.0)
    if start_min == 0 and end_min == 1440:
        return now

    # Non-wrapping window: [start, end)
    if start_min < end_min:
        if current < start_min:
            return _dt_at_minutes(now, minutes=start_min)
        # current >= end_min (or between end and start, but that's outside)
        return _dt_at_minutes(now, minutes=start_min) + timedelta(days=1)

    # Wrapping window: [start, 24:00) U [0, end)
    # Outside means end <= current < start -> next start is today at start.
    return _dt_at_minutes(now, minutes=start_min)


def _load_runtime_config() -> Tuple[RuntimeConfig, dict[str, str]]:
    provider = os.environ.get("PROVIDER", "twelvedata").strip().lower()
    poll_seconds = _parse_int(os.environ.get("POLL_SECONDS"), 60)
    cooldown_minutes = _parse_int(os.environ.get("COOLDOWN_MINUTES"), 120)
    dry_run = _parse_bool(os.environ.get("DRY_RUN"), False)

    symbol = os.environ.get("TWELVEDATA_SYMBOL", "XAU/USD")
    twelvedata_timezone = os.environ.get("TWELVEDATA_TIMEZONE", "").strip()
    state_file = os.environ.get("STATE_FILE", ".goldx_state.json").strip()
    monitor = _load_monitor_window()

    low_window_min = _parse_int(os.environ.get("LOW_WINDOW_MIN"), 60)
    slope_window_min = _parse_int(os.environ.get("SLOPE_WINDOW_MIN"), 10)
    quantile_window_min = _parse_int(os.environ.get("QUANTILE_WINDOW_MIN"), 240)
    q_low = _parse_float(os.environ.get("Q_LOW"), 25)
    q_skip = _parse_float(os.environ.get("Q_SKIP"), 60)
    stall_min = _parse_int(os.environ.get("STALL_MIN"), 4)
    micro_rebound_pct = _parse_float(os.environ.get("MICRO_REBOUND_PCT"), 0.03)
    rebound_pct = _parse_float(os.environ.get("REBOUND_PCT"), 0.15)
    rebound_max_pct = _parse_float(os.environ.get("REBOUND_MAX_PCT"), 0.5)
    low_max_age_min = _parse_int(os.environ.get("LOW_MAX_AGE_MIN"), 20)
    slope_threshold = _parse_float(os.environ.get("SLOPE_THRESHOLD_PCT_PER_MIN"), -0.005)
    hist_rise_bars = _parse_int(os.environ.get("HIST_RISE_BARS"), 3)
    alert_plot_on_email = _parse_bool(os.environ.get("ALERT_PLOT_ON_EMAIL"), True)
    alert_plot_dir = os.environ.get("ALERT_PLOT_DIR", "alert_out").strip() or "alert_out"

    cfg = RuntimeConfig(
        provider=provider,
        poll_seconds=poll_seconds,
        cooldown_minutes=cooldown_minutes,
        dry_run=dry_run,
        symbol=symbol,
        twelvedata_timezone=twelvedata_timezone,
        state_file=state_file,
        monitor=monitor,
        alert_plot_on_email=alert_plot_on_email,
        alert_plot_dir=alert_plot_dir,
        low_window_min=low_window_min,
        slope_window_min=slope_window_min,
        quantile_window_min=quantile_window_min,
        q_low=q_low,
        q_skip=q_skip,
        stall_min=stall_min,
        micro_rebound_pct=micro_rebound_pct,
        rebound_pct=rebound_pct,
        rebound_max_pct=rebound_max_pct,
        low_max_age_min=low_max_age_min,
        slope_threshold_pct_per_min=slope_threshold,
        hist_rise_bars=hist_rise_bars,
    )

    secrets: dict[str, str] = {}
    if provider == "twelvedata":
        secrets["TWELVEDATA_API_KEYS"] = os.environ.get("TWELVEDATA_API_KEYS", "").strip()
        secrets["TWELVEDATA_API_KEY"] = os.environ.get("TWELVEDATA_API_KEY", "").strip()
    return cfg, secrets


def _load_email_config() -> EmailConfig:
    host = os.environ.get("SMTP_HOST", "").strip()
    port = _parse_int(os.environ.get("SMTP_PORT"), 465)
    username = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "").strip()
    from_addr = os.environ.get("SMTP_FROM", username).strip()
    to_raw = os.environ.get("SMTP_TO", "").strip()
    to_addrs = [a.strip() for a in to_raw.split(",") if a.strip()]

    use_ssl = _parse_bool(os.environ.get("SMTP_SSL"), True)
    use_starttls = _parse_bool(os.environ.get("SMTP_STARTTLS"), False)
    if use_ssl and use_starttls:
        raise ValueError("请只启用 SMTP_SSL 或 SMTP_STARTTLS 其中一个")

    if not host or not username or not password or not from_addr or not to_addrs:
        raise ValueError("缺少邮箱配置：SMTP_HOST/SMTP_USER/SMTP_PASSWORD/SMTP_FROM/SMTP_TO")

    return EmailConfig(
        host=host,
        port=port,
        username=username,
        password=password,
        from_addr=from_addr,
        to_addrs=to_addrs,
        use_ssl=use_ssl,
        use_starttls=use_starttls,
    )


def _format_email_body(
    *,
    now: datetime,
    symbol: str,
    points: list[PricePoint],
    signal: Signal,
    tz: tzinfo,
) -> str:
    last_point = points[-1]
    now_tz = now.astimezone(tz)
    ts_str = last_point.ts.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "【黄金买入提醒（动能减弱）】",
        f"时间：{now_tz.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"数据源：{symbol}（1min）",
        f"最新：{signal.metrics.get('last', float('nan')):.6f}",
        f"近{int(os.environ.get('LOW_WINDOW_MIN','60'))}分钟最低：{signal.metrics.get('low', float('nan')):.6f}",
        f"反弹幅度：{signal.metrics.get('rebound_pct', float('nan')):.3f}%",
        f"近{int(os.environ.get('SLOPE_WINDOW_MIN','10'))}分钟斜率：{signal.metrics.get('slope_pct_per_min', float('nan')):.4f}%/min",
        f"理由：{signal.reason}",
        "",
        "说明：这是基于短期走势的概率判断，无法保证抄到底。建议分批买入/控制仓位。",
        f"(最近一根K线时间戳：{ts_str})",
    ]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    load_dotenv()
    once = "--once" in argv
    test_email = "--test-email" in argv
    backtest = "--backtest" in argv
    show_help = ("-h" in argv) or ("--help" in argv)

    cfg, secrets = _load_runtime_config()

    if show_help:
        print(
            "用法：\n"
            "  监测并发邮件：python3 goldx_alert.py\n"
            "  只跑一次：  python3 goldx_alert.py --once\n"
            "  测试邮箱：  python3 goldx_alert.py --test-email\n"
            "  回测（模拟）：python3 goldx_alert.py --backtest [--date YYYY-MM-DD] [--start HH:MM] [--end HH:MM]\n"
            "\n"
            "回测常用参数：\n"
            "  --outdir backtest_out     输出目录（默认 backtest_out）\n"
            "  --outputsize 5000         拉取K线条数（默认 5000）\n"
            "  --warmup-min 0            计算预热分钟数（默认 0）\n"
            "  --no-plot                 不生成图片\n"
            "  --refresh-cache           忽略缓存，强制重新拉取数据\n"
            "  --no-cache                不读写缓存\n"
        )
        return 0

    if test_email:
        try:
            email_cfg = _load_email_config()
        except Exception as e:
            print(f"邮箱配置错误：{e}", file=sys.stderr)
            return 2
        now = datetime.now(cfg.monitor.tz)
        subject = "goldx-alert 测试邮件"
        body = (
            "这是一封测试邮件，用于验证 SMTP 配置是否可用。\n"
            f"时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
        )
        if cfg.dry_run:
            print("DRY_RUN=1：将发送测试邮件但已跳过\n" + body)
        else:
            send_email(email_cfg, subject=subject, body=body)
            print("已发送测试邮件")
        return 0

    if cfg.provider != "twelvedata":
        print(f"暂仅支持 PROVIDER=twelvedata；当前={cfg.provider}", file=sys.stderr)
        return 2

    api_keys = _parse_api_keys(secrets.get("TWELVEDATA_API_KEYS")) or _parse_api_keys(secrets.get("TWELVEDATA_API_KEY"))
    if not api_keys:
        print("缺少 TWELVEDATA_API_KEYS 或 TWELVEDATA_API_KEY（建议放到 .env）", file=sys.stderr)
        return 2
    api_key_index = 0

    def _is_key_likely_bad(err: Exception) -> bool:
        msg = str(err).lower()
        keywords = [
            "limit",
            "quota",
            "credits",
            "rate",
            "too many",
            "429",
            "unauthorized",
            "forbidden",
            "invalid",
            "apikey",
            "api key",
        ]
        return any(k in msg for k in keywords)

    def fetch_points_with_key_failover(
            api_keys: list[str],
            cfg: RuntimeConfig, 
            outputsize: int,
            api_key_index: int = 0,
            ) -> tuple[list[PricePoint], int]:
        last_err: Optional[Exception] = None
        for offset in range(len(api_keys)):
            idx = (api_key_index + offset) % len(api_keys)
            key = api_keys[idx]
            try:
                points = fetch_twelvedata_1min_series(
                    api_key=key,
                    symbol=cfg.symbol,
                    interval="1min",
                    outputsize=outputsize,
                    tz=cfg.monitor.tz,
                )
                # Success: move to next key on next poll to spread quota usage.
                next_idx = (idx + 1) % len(api_keys)
                return points, next_idx
            except Exception as e:
                last_err = e
                if _is_key_likely_bad(e):
                    print(f"{now.strftime('%H:%M:%S %Z')},API_KEY{idx} 可能受限/失效，：{e}", file=sys.stderr)
                    continue
                # 非配额/鉴权类错误也尝试下一个 key（可能是临时网络问题）
                print(f"{now.strftime('%H:%M:%S %Z')},API_KEY{idx} 请求失败，尝试切换：{e}", file=sys.stderr)
                continue
        assert last_err is not None
        raise last_err

    def _handle_backtest(argv: list[str], cfg: RuntimeConfig, api_keys: list[str]) -> int:
        """处理回测逻辑"""
        date_str = _get_arg_value(argv, "--date")
        start_str = _get_arg_value(argv, "--start", "09:00") or "09:00"
        end_str = _get_arg_value(argv, "--end", "22:00") or "22:00"
        out_root = _get_arg_value(argv, "--outdir", "backtest_out") or "backtest_out"
        outputsize = int(_get_arg_value(argv, "--outputsize", "5000") or "5000")
        warmup_min = int(_get_arg_value(argv, "--warmup-min", "0") or "0")
        do_plot = "--no-plot" not in argv
        refresh_cache = "--refresh-cache" in argv
        no_cache = "--no-cache" in argv
        cache_root = _get_arg_value(argv, "--cache-dir", "backtest_cache") or "backtest_cache"

        tz = cfg.monitor.tz
        bt_date = _parse_date(date_str) if date_str else _today_in_tz(tz)
        start_min = _parse_hhmm_to_minutes(start_str)
        end_min = _parse_hhmm_to_minutes(end_str)
        base = datetime(bt_date.year, bt_date.month, bt_date.day, 0, 0, 0, tzinfo=tz)
        eval_start = base + timedelta(minutes=start_min)
        # Include the candle at END time.
        if end_min == 1440:
            eval_end_excl = base + timedelta(days=1)
        else:
            eval_end_excl = base + timedelta(minutes=end_min) + timedelta(minutes=1)
        warmup_start = eval_start - timedelta(minutes=max(0, warmup_min))

        symbol_slug = _sanitize_symbol_for_path(cfg.symbol)
        out_dir = os.path.join(out_root, symbol_slug, bt_date.isoformat())
        os.makedirs(out_dir, exist_ok=True)

        points: Optional[list[PricePoint]] = None
        
        cache_path = os.path.join(cache_root, symbol_slug, f"{bt_date.isoformat()}.json")
        if not no_cache and not refresh_cache:
            points = _load_points_cache(cache_path, tz=tz)
            if points:
                print(f"已从缓存加载数据：{cache_path}")

        if not points:
            now = datetime.now(tz)
            api_key_index = 0
            print(f"拉取K线：symbol={cfg.symbol} interval=1min outputsize={outputsize}")
            points,api_key_index = fetch_points_with_key_failover(api_keys,cfg,outputsize)
            if not no_cache:
                _save_points_cache(cache_path, points,tz=cfg.monitor.tz_name)
                print(f"已写入缓存：{cache_path}")

 
        points = [p for p in points if warmup_start <= p.ts < eval_end_excl]
        points.sort(key=lambda p: p.ts)
        eval_points = [p for p in points if eval_start <= p.ts < eval_end_excl]
        if len(eval_points) < 10:
            print(
                "回测范围内数据太少，可能是 outputsize 不够或数据源缺失。\n"
                f"范围：{eval_start.isoformat()} ~ {eval_end_excl.isoformat()}  (points={len(eval_points)})\n"
                f"建议：提高 --outputsize，或换一个日期/品种。",
                file=sys.stderr,
            )
            return 2


        window_size = max(
            80,
            cfg.low_window_min,
            cfg.slope_window_min,
            cfg.quantile_window_min,
            cfg.stall_min,
        )
        cooldown = timedelta(minutes=max(0, cfg.cooldown_minutes))

        alerts: list[BacktestAlert] = []
        low_so_far: Optional[PricePoint] = None
        last_alert_ts: Optional[datetime] = None
        seen: list[PricePoint] = []

        for p in points:
            if p.ts >= eval_end_excl:
                break
            seen.append(p)
            if p.ts < eval_start:
                continue
            if low_so_far is None or p.price < low_so_far.price:
                low_so_far = p

            window = seen[-window_size:]
            signal = compute_buy_signal(
                window,
                low_window_min=cfg.low_window_min,
                slope_window_min=cfg.slope_window_min,
                quantile_window_min=cfg.quantile_window_min,
                q_low=cfg.q_low,
                q_skip=cfg.q_skip,
                stall_min=cfg.stall_min,
                micro_rebound_pct=cfg.micro_rebound_pct,
                rebound_pct=cfg.rebound_pct,
                rebound_max_pct=cfg.rebound_max_pct,
                low_max_age_min=cfg.low_max_age_min,
                slope_threshold_pct_per_min=cfg.slope_threshold_pct_per_min,
                hist_rise_bars=cfg.hist_rise_bars,
            )
            if not signal.should_alert:
                continue
            if last_alert_ts is not None and (p.ts - last_alert_ts) < cooldown:
                continue

            low_window = _min_price_point(window[-cfg.low_window_min:])
            assert low_so_far is not None
            alert = BacktestAlert(
                ts=p.ts,
                price=p.price,
                low_window_ts=low_window.ts,
                low_window_price=low_window.price,
                low_so_far_ts=low_so_far.ts,
                low_so_far_price=low_so_far.price,
                reason=signal.reason,
                metrics=signal.metrics,
            )
            alerts.append(alert)
            last_alert_ts = p.ts

            if do_plot:
                fn = f"alert_{bt_date.isoformat()}_{p.ts.strftime('%H%M')}.png"
                ok = _plot_alert_snapshot(
                    points=list(seen),
                    alert=alert,
                    out_path=os.path.join(out_dir, fn),
                    title=f"{cfg.symbol} backtest alert {p.ts.strftime('%Y-%m-%d %H:%M %Z')}",
                    tz=tz,
                )
                if not ok:
                    do_plot = False
                    print("未检测到 matplotlib，已跳过绘图（可先安装 matplotlib 再跑回测）。", file=sys.stderr)

        # Write outputs
        header = [
            "alert_time",
            "price",
            "low_window_time",
            "low_window_price",
            "low_so_far_time",
            "low_so_far_price",
            "rebound_pct",
            "slope_pct_per_min",
            "macd_hist_last",
            "reason",
        ]
        rows: list[list[Any]] = []
        for a in alerts:
            rows.append(
                [
                    a.ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    a.price,
                    a.low_window_ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    a.low_window_price,
                    a.low_so_far_ts.strftime("%Y-%m-%d %H:%M:%S %Z"),
                    a.low_so_far_price,
                    a.metrics.get("rebound_pct"),
                    a.metrics.get("slope_pct_per_min"),
                    a.metrics.get("macd_hist_last"),
                    a.reason,
                ]
            )
        csv_path = os.path.join(out_dir, "alerts.csv")
        _write_csv(csv_path, header, rows)

        summary = {
            "symbol": cfg.symbol,
            "tz": cfg.monitor.tz_name,
            "date": bt_date.isoformat(),
            "start": start_str,
            "end": end_str,
            "warmup_min": warmup_min,
            "data_range": {
                "first": points[0].ts.isoformat(),
                "last": points[-1].ts.isoformat(),
                "eval_first": eval_points[0].ts.isoformat(),
                "eval_last": eval_points[-1].ts.isoformat(),
            },
            "points_in_range": len(eval_points),
            "alerts": len(alerts),
            "window_size": window_size,
            "params": {
                "LOW_WINDOW_MIN": cfg.low_window_min,
                "SLOPE_WINDOW_MIN": cfg.slope_window_min,
                "QUANTILE_WINDOW_MIN": cfg.quantile_window_min,
                "Q_LOW": cfg.q_low,
                "Q_SKIP": cfg.q_skip,
                "STALL_MIN": cfg.stall_min,
                "MICRO_REBOUND_PCT": cfg.micro_rebound_pct,
                "REBOUND_PCT": cfg.rebound_pct,
                "REBOUND_MAX_PCT": cfg.rebound_max_pct,
                "LOW_MAX_AGE_MIN": cfg.low_max_age_min,
                "SLOPE_THRESHOLD_PCT_PER_MIN": cfg.slope_threshold_pct_per_min,
                "HIST_RISE_BARS": cfg.hist_rise_bars,
            "COOLDOWN_MINUTES": cfg.cooldown_minutes,
        },
    }


        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        if do_plot:
            full_points = [p for p in points if eval_start <= p.ts < eval_end_excl]
            ok = _plot_full_day(
                points=full_points,
                alerts=alerts,
                out_path=os.path.join(out_dir, "full.png"),
                title=f"{cfg.symbol} backtest {bt_date.isoformat()} ({start_str}-{end_str} {cfg.monitor.tz_name})",
                tz=tz,
            )
            if not ok:
                print("未检测到 matplotlib，已跳过绘图（可先安装 matplotlib 再跑回测）。", file=sys.stderr)

        low_day = _min_price_point(eval_points)
        print(f"回测完成：输出目录={out_dir}")
        print(f"- 区间最低：{low_day.ts.strftime('%Y-%m-%d %H:%M %Z')} price={low_day.price}")
        print(f"- 提醒次数：{len(alerts)}（详情见 {csv_path}）")
        return 0

    if backtest:
        return _handle_backtest(argv, cfg, api_keys)

    try:
        email_cfg = _load_email_config()
    except Exception as e:
        print(f"邮箱配置错误：{e}", file=sys.stderr)
        return 2

    last_alert_at: Optional[datetime] = None
    if cfg.state_file:
        try:
            if os.path.exists(cfg.state_file):
                with open(cfg.state_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and raw.get("last_alert_at"):
                    last_alert_at = datetime.fromisoformat(raw["last_alert_at"]).replace(tzinfo=cfg.monitor.tz)
        except Exception as e:
            print(f"读取状态失败（将忽略）：{e}", file=sys.stderr)

    def can_alert(now: datetime) -> bool:
        nonlocal last_alert_at
        if last_alert_at is None:
            return True
        return now - last_alert_at >= timedelta(minutes=cfg.cooldown_minutes)
    
    api_key_index = 0
    idle_until: Optional[datetime] = None

    while True:
        now = datetime.now(cfg.monitor.tz)
        if not _is_within_window(now, cfg.monitor):
            next_start = _next_window_start(now, cfg.monitor)
            api_key_index = 0
            if idle_until is None or next_start != idle_until:
                print(
                    f"[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] 不在监测时段 "
                    f"({cfg.monitor.tz_name} {_format_minutes(cfg.monitor.start_min)}"
                    f"-{_format_minutes(cfg.monitor.end_min)})，"
                    f"休眠至 {next_start.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                )
                idle_until = next_start
            if once:
                return 0
            sleep_s = (next_start - now).total_seconds()
            time.sleep(max(5, min(sleep_s, 3600)))
            continue

        idle_until = None
        try:
            outputsize = max(
                80,
                cfg.low_window_min + 30,
                cfg.slope_window_min + 5,
                cfg.quantile_window_min + 5,
                cfg.stall_min + 5,
            )
            points, api_key_index = fetch_points_with_key_failover(api_keys, cfg, outputsize, api_key_index)
            
            signal = compute_buy_signal(
                points,
                low_window_min=cfg.low_window_min,
                slope_window_min=cfg.slope_window_min,
                quantile_window_min=cfg.quantile_window_min,
                q_low=cfg.q_low,
                q_skip=cfg.q_skip,
                stall_min=cfg.stall_min,
                micro_rebound_pct=cfg.micro_rebound_pct,
                rebound_pct=cfg.rebound_pct,
                rebound_max_pct=cfg.rebound_max_pct,
                low_max_age_min=cfg.low_max_age_min,
                slope_threshold_pct_per_min=cfg.slope_threshold_pct_per_min,
                hist_rise_bars=cfg.hist_rise_bars,
            )
            print(
                f"[{now.strftime('%H:%M:%S %Z')}] key={api_key_index+1}/{len(api_keys)} last={signal.metrics.get('last')} "
                f"rebound={signal.metrics.get('rebound_pct')}% slope={signal.metrics.get('slope_pct_per_min')}%/min "
                f"alert={signal.should_alert}"
            )

            if signal.should_alert and can_alert(now):
                subject = f"黄金提醒：下跌动能减弱（{cfg.symbol}）"
                body = _format_email_body(now=now, symbol=cfg.symbol, points=points, signal=signal, tz=cfg.monitor.tz)
                attachments: list[str] = []
                if cfg.alert_plot_on_email:
                    try:
                        plot_path = _render_realtime_alert_plot(points, cfg, signal)
                        if plot_path:
                            attachments.append(plot_path)
                    except Exception as e:
                        print(f"生成提醒图失败（将继续发送邮件）：{e}", file=sys.stderr)
                if cfg.dry_run:
                    msg = "DRY_RUN=1：将发送邮件但已跳过\n" + body
                    if attachments:
                        msg += f"\n(附件：{', '.join(attachments)})"
                    print(msg)
                else:
                    send_email(email_cfg, subject=subject, body=body, attachments=attachments)
                    print("已发送提醒邮件")
                last_alert_at = now
                if cfg.state_file:
                    try:
                        with open(cfg.state_file, "w", encoding="utf-8") as f:
                            json.dump({"last_alert_at": now.isoformat()}, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"写入状态失败（将忽略）：{e}", file=sys.stderr)
        except Exception as e:
            print(f"[{now.strftime('%H:%M:%S %Z')}] 获取/计算失败：{e}", file=sys.stderr)

        if once:
            return 0
        time.sleep(max(5, cfg.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
