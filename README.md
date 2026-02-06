# goldx-alert

分钟级“下跌动能减弱”提醒（邮件版）。注意：这是概率判断，不保证抄到最低点。

## 快速开始

1. 复制 `config.example.env` 为 `.env`，填好：
   - `TWELVEDATA_API_KEYS`（或单个 `TWELVEDATA_API_KEY`）
   - `SMTP_*`（用邮箱的授权码/应用专用密码）
   - 多个 API key 用英文逗号分隔，程序会轮询使用并在配额/限流时自动切换
2. 运行：

```bash
python3 goldx_alert.py
```

先测试邮件是否能发出去：

```bash
python3 goldx_alert.py --test-email
```

> 如果你用 Twelve Data 免费额度，建议把 `POLL_SECONDS` 设为 `120` 或更大，或只在你关注的时段运行。

只跑一次（适合手动/定时任务）：

```bash
python3 goldx_alert.py --once
```

## 回测（模拟）+ 自动出图

用“只看当下与过去，不看未来”的方式，把某天某个时间段逐分钟模拟跑一遍，输出程序会在哪些点提醒。

```bash
python3 goldx_alert.py --backtest --date 2026-02-05 --start 09:00 --end 22:00
```

也可以用本地 CSV 回测（列含 `datetime/ts` + `price/close`，时间按 `MONITOR_TZ` 解释）：

```bash
python3 goldx_alert.py --backtest --csv data.csv --date 2026-02-05 --start 09:00 --end 22:00
```

输出默认在 `backtest_out/<symbol>/<date>/`：

- `alerts.csv`：每次提醒的时间/价格/指标/理由
- `summary.json`：本次回测的参数与统计
- `full.png`：整段区间价格 + 所有提醒点
- `alert_*.png`：每次提醒触发时的“截至当时”的折线图快照（圈出关键点）

> 若提示没有 matplotlib，可先安装：`python3 -m pip install matplotlib`
>
> 回测默认会缓存拉取到的数据到 `backtest_cache/`，反复调参时不会重复消耗 API 请求；需要强制刷新可加 `--refresh-cache`。
>
> 如果你发现回测曲线和 TradingView（UTC+8）时间对不上：
> - 程序会尝试根据“最新K线时间”自动推断 TwelveData 的源时区，并对旧缓存做一次性修复；
> - 仍可在 `.env` 里设置 `TWELVEDATA_TIMEZONE=Asia/Shanghai`（请求 API 直接返回 UTC+8），或用 `TWELVEDATA_SOURCE_TZ=UTC+11` 这类值强制指定源时区；必要时加 `--refresh-cache` 重新拉取数据。

## 只在 9:00-24:00 监测（中国大陆时间）

在 `.env` 里设置：

- `MONITOR_TZ=Asia/Shanghai`
- `MONITOR_START=09:00`
- `MONITOR_END=24:00`

## 参数怎么调（核心参数）

- `REBOUND_PCT`：从“最近低点”反弹到多少开始考虑提醒（默认 0.15%）
- `REBOUND_MAX_PCT`：反弹上限，超过就认为追高不提醒（默认 0.5%，设为 0 可关闭）
- `LOW_MAX_AGE_MIN`：低点需是“最近多少分钟内”的低（默认 20，设为 0 可关闭）
- `SLOPE_THRESHOLD_PCT_PER_MIN`：近 `SLOPE_WINDOW_MIN` 分钟的斜率阈值（越大越保守）
- `COOLDOWN_MINUTES`：发过一次后多久不再提醒（防止震荡刷屏）

## 分位过滤（更贴近低点、减少追高）

- `QUANTILE_WINDOW_MIN`：滚动分位窗口长度（默认 240 分钟）
- `Q_LOW`：低分位阈值（默认 25，<=该分位可触发“提前提醒”）
- `Q_SKIP`：高分位阈值（默认 60，>该分位直接不提醒）
- `STALL_MIN`：停跌确认分钟数（默认 4；最近 N 分钟不再创新低）
- `MICRO_REBOUND_PCT`：低分位“微反弹”阈值（默认 0.03%）

## 邮件提醒附图

- `ALERT_PLOT_ON_EMAIL`：实盘提醒邮件是否附带本次 alert 的图（默认 1）
- `ALERT_PLOT_DIR`：图片输出目录（默认 `alert_out`）
