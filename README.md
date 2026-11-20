# 直播多轨变采样的连续卷积与时间交接

`samples/capture.json` 是离线合成的交接采集，包含原始数据和接收顺序。[交接协议](交接协议.md) 描述各字段及上下游共同使用的数值口径。`read_capture` 可读取采集文件并列出操作顺序。

`resampler.py` 是流式 FIR 变采样与多轨时间交接引擎：按事件接收 `clock` / `chunk` / `gap` / `safe`，用 `preview` 给字幕模型可暂用的值、已知系数部分和与缺口状态，用 `publish` 只交付双轨稳定且支撑已定的连续前缀（断流点交付空档）。已交付区间不可再被修订、可随时回取；每轨热历史只保留 `retention` 个输入编号跨度，越界且不能安全修正时明确请求重送或拒绝。`replay.py` 把采集回放进引擎并打印每步裁决。

环境使用 Python 3.11 和锁定的 NumPy。首次在本地准备虚拟环境后运行以下命令。

```bash
python -m pip install -r requirements.txt
python -m read_capture samples/capture.json
python -m replay samples/capture.json
python -m unittest discover -s tests -v
```

