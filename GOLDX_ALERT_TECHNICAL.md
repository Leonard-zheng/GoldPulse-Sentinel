/。爱的是 # goldx_alert.py 技术文档

## 1. 文件定位与目标

`goldx_alert.py` 是项目的单文件核心程序，承担三类职责：

1. 实盘轮询行情并发送买入提醒邮件。
2. 回测指定日期/时段，输出提醒记录、统计和图像。
3. 通过同一套策略参数在“回测/实盘”之间保持一致的信号判定逻辑。

核心设计目标：

- 使用 1 分钟级别行情，识别“下跌动能减弱 + 低位反转”。
- 尽量减少追高提醒（分位过滤、反弹上限、低点新鲜度）。
- 保留可解释的触发原因（`reason` + `metrics`）。

---

## 2. 运行模式

`main(argv)` 中支持四种运行模式：

1. `--help`：打印帮助并退出。
2. `--test-email`：仅测试 SMTP 发送。
3. `--backtest`：进入回测流程 `_handle_backtest(...)` 并退出。
4. 默认：进入实盘循环 `while True`，按 `POLL_SECONDS` 轮询。

辅助控制参数：

- `--once`：只跑一次轮询后退出（适合调试/cron）。

---

## 3. 代码结构总览

### 3.1 数据结构

- `PricePoint(ts, price)`：单个行情点。
- `Signal(should_alert, reason, metrics)`：信号结果。
- `BacktestAlert(...)`：回测和实时附图复用的告警快照结构。
- `EmailConfig`：SMTP 配置。
- `RuntimeConfig`：运行时所有策略与系统参数。
- `MonitorWindow`：监控时段定义（时区 + 起止分钟）。

### 3.2 核心模块分层

1. 基础解析与工具函数（时间、参数、路径）。
2. 数据源访问（TwelveData 拉取 + key 轮询切换）。
3. 指标与策略（EMA/MACD/斜率/分位 + 信号组合）。
4. 回测流程（逐分钟模拟、缓存、导出、绘图）。
5. 实盘流程（轮询、冷却、邮件、状态持久化）。

---

## 4. 数据拉取与 API Key 管理

## 4.1 TwelveData 客户端复用

- 使用进程内全局 `_TD_CLIENTS: dict[str, TDClient]` 缓存客户端。
- 同一 API Key 不重复创建 `TDClient`，避免每次轮询重建会话。
- 进程重启后缓存自然清空（属于正常行为）。

## 4.2 多 Key 轮询 + 失败切换

`fetch_points_with_key_failover(...)` 的行为：

1. 从当前 `api_key_index` 开始尝试请求。
2. 请求成功后，返回数据并把“下次起始 key”设为下一个索引（轮询分摊额度）。
3. 若请求失败（额度/鉴权/网络），继续尝试下一个 key。
4. 全部 key 失败时抛出最后一个异常。

说明：

- 这是“轮询 + failover”组合，不是只在故障时才换 key。
- 仍受总额度、网络、上游服务状态约束，不能数学上保证“永不耗尽”。

---

## 5. 策略引擎 `compute_buy_signal(...)`

## 5.1 输入与最小样本

- 输入是按时间升序的 `points`。
- 最小样本门槛：`len(points) >= max(40, LOW_WINDOW_MIN, SLOPE_WINDOW_MIN)`。

## 5.2 核心特征

1. **低点与反弹**
   - 在最近 `LOW_WINDOW_MIN` 内找低点 `low_point`。
   - `rebound_pct = (last - low) / low * 100`。
   - `low_age_min` 表示低点距当前的分钟数。

2. **分位过滤（滚动窗口）**
   - 用最近 `QUANTILE_WINDOW_MIN` 个价格计算分位；
   - 若不足窗口长度，使用当前已有全部样本。
   - 计算 `q_low_value` 和 `q_skip_value`。

3. **趋势强度**
   - 近 `SLOPE_WINDOW_MIN` 的线性回归斜率，归一化为 `%/min`。

4. **MACD 柱体连续回升**
   - 用 EMA 计算 MACD Histogram。
   - 判断最近 `HIST_RISE_BARS` 根是否严格递增。

5. **停跌与微反弹**
   - `STALL_MIN`：最近 N 分钟不再创新低。
   - `MICRO_REBOUND_PCT`：低分位提前提醒的最小微反弹阈值。

## 5.3 决策分层

1. **高位过滤层**
   - `last > q_skip_value` 直接 `should_alert=False`。

2. **低分位提前层（`last <= q_low_value`）**
   - 条件：低点新鲜度 + 停跌 + 微反弹 + 反弹上限 + 斜率达标。
   - 不强制依赖 MACD 连续回升（目的是更早捕捉）。

3. **中分位确认层（`q_low < last <= q_skip`）**
   - 条件：反弹区间、低点新鲜度、斜率达标、MACD 连升。

该函数会输出详细 `reason` 与 `metrics`，便于回测解释和调参。

---

## 6. 回测流程 `_handle_backtest(...)`

## 6.1 时段与样本准备

1. 解析回测参数（日期、时段、输出目录、是否绘图、缓存控制）。
2. 计算 `eval_start` 与 `eval_end_excl`（`end` 时刻 K 线被包含）。
3. 优先读取 `backtest_cache`；失败则走 API 拉取并写缓存。
4. 截取回测区间样本并排序。

## 6.2 逐分钟模拟（无未来函数）

每推进一个点 `p`：

1. 仅使用当前时刻之前的 `seen` 数据构建窗口；
2. 调用 `compute_buy_signal(window, ...)`；
3. 通过冷却 `COOLDOWN_MINUTES` 过滤过密提醒；
4. 记录 `BacktestAlert`，可选输出当时快照图。

这保证信号判定只依赖“当时可见信息”。

## 6.3 输出产物

目录：`backtest_out/<symbol>/<date>/`

- `alerts.csv`：提醒明细。
- `summary.json`：参数、样本范围、提醒总数等。
- `full.png`：全时段曲线 + 提醒点。
- `alert_*.png`：每次提醒时的局部快照。

---

## 7. 实盘流程（默认模式）

## 7.1 启动与状态恢复

1. 读取 SMTP 配置。
2. 从 `STATE_FILE` 恢复 `last_alert_at`（用于重启后继续冷却控制）。

## 7.2 监控时段控制

- 不在监控窗口时，计算下个开始时间并休眠。
- 在窗口内才拉取行情并计算信号。

## 7.3 轮询一次的执行顺序

1. 计算 `outputsize`（保证策略窗口有足够历史）。
2. 拉取最新 1 分钟序列（多 key 轮询）。
3. 计算信号。
4. 若满足信号且通过冷却：
   - 组装邮件正文；
   - 可选生成实时附图（`ALERT_PLOT_ON_EMAIL=1`）；
   - 发送邮件；
   - 更新并落盘 `last_alert_at`。

## 7.4 重启影响

- 会丢失：进程内临时变量（客户端缓存、key 索引、idle 标记）。
- 会保留：`STATE_FILE` 中的 `last_alert_at`，因此冷却策略可跨重启延续。

---

## 8. 邮件与附图

## 8.1 邮件发送

- 支持 SSL 或 STARTTLS（二选一）。
- 支持附件列表，自动推断 MIME 类型。

## 8.2 实盘附图

- 触发提醒时调用 `_render_realtime_alert_plot(...)`；
- 输出目录：`ALERT_PLOT_DIR/<symbol>/<date>/alert_*.png`；
- 附件随邮件发送，便于快速查看触发上下文。

---

## 9. 配置项分组（来自 `.env`）

## 9.1 数据源与轮询

- `PROVIDER`（当前仅 `twelvedata`）
- `TWELVEDATA_API_KEYS` / `TWELVEDATA_API_KEY`
- `TWELVEDATA_SYMBOL`
- `POLL_SECONDS`

## 9.2 监控窗口

- `MONITOR_TZ`
- `MONITOR_START`
- `MONITOR_END`

## 9.3 策略参数

- `LOW_WINDOW_MIN`
- `SLOPE_WINDOW_MIN`
- `QUANTILE_WINDOW_MIN`
- `Q_LOW`
- `Q_SKIP`
- `STALL_MIN`
- `MICRO_REBOUND_PCT`
- `REBOUND_PCT`
- `REBOUND_MAX_PCT`
- `LOW_MAX_AGE_MIN`
- `SLOPE_THRESHOLD_PCT_PER_MIN`
- `HIST_RISE_BARS`
- `COOLDOWN_MINUTES`

## 9.4 邮件与状态

- `SMTP_*`
- `STATE_FILE`
- `DRY_RUN`
- `ALERT_PLOT_ON_EMAIL`
- `ALERT_PLOT_DIR`

---

## 10. 日志与运维

当前日志机制是“标准输出重定向”：

- 启动脚本 `start_goldx.sh` 使用 `>> logs/goldx.log 2>&1`。
- 代码中的 `print(...)` 与 `stderr` 都写入同一日志文件。

常用命令：

```bash
tail -f logs/goldx.log
```

---

## 11. 异常处理与边界行为

1. API 拉取失败会切 key，全部失败后打印错误并继续下一轮。
2. 数据不足时信号返回“数据不足”，不触发提醒。
3. SMTP/绘图异常不会让主循环直接崩溃（多数情况下被捕获并打印）。
4. 缓存读取失败会回退到实时拉取。
5. `Q_SKIP < Q_LOW` 会在内部自动纠正为 `Q_SKIP = Q_LOW`。

---

## 12. 可改进方向（工程视角）

1. 以 `logging` 替代 `print`，加入日志级别与滚动切分。
2. 增加健康检查与告警（如连续 N 次 API 失败）。
3. 将配置校验前置，启动时一次性报错更清晰。
4. 为策略函数增加单元测试（分位边界、停跌判断、冷却判定）。
5. 将回测与实盘策略接口再抽象一层，降低 `main` 的体积。

