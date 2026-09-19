# 字幕断句(Subtitle Segmenter)

基于 FastAPI + 原生 HTML/JavaScript 的字幕断句网页应用。用户粘贴带时间戳的词序列,可标记不可拆短语;系统在「每条最多两行、行宽与阅读速度限制」下,用全局动态规划选择分段,使 **镜头切点偏差 + 行长不均 + 不自然停顿** 的加权总代价最低。页面实时展示断句预览与逐段代价,调整权重即时重算;无可行方案时定位首个冲突区间及原因。

## 启动

```bash
docker compose up --build
```

依赖安装与应用初始化均随容器启动自动完成,无需本地环境。

## 访问

- 网页界面: http://localhost:8000
- API: `POST http://localhost:8000/api/segment`

## 配置

无需环境变量;所有参数在网页左侧面板调整(也可通过 API 的 `config` 字段传入):

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `max_chars_per_line` | 42 | 单行最大字符数(硬约束,每段最多 2 行) |
| `max_cps` | 20 | 最大阅读速度,字/秒(硬约束) |
| `min_duration` | 0.8 | 每段最短时长,秒(硬约束) |
| `max_duration` | 7.0 | 每段最长时长,秒(硬约束) |
| `pause_threshold` | 0.3 | 词间间隔小于该值视为不自然断点,秒 |
| `shot_dev_cap` | 0.5 | 镜头切点偏差归一化上限,秒 |
| `crossing_penalty` | 1.0 | 分段跨越一个镜头切点的惩罚 |
| `w_shot` | 1.0 | 权重:镜头切点偏差 |
| `w_imbalance` | 1.0 | 权重:行长不均 |
| `w_pause` | 1.0 | 权重:不自然停顿 |

## 输入格式

- **词序列**:每行 `开始 结束 文本`,时间支持 `HH:MM:SS.mmm` 或秒数;也接受 JSON 数组 `[{"text","start","end"}]`。
- **不可拆短语**:每行一个,词下标区间(如 `3-6`,含端点)或短语原文(自动匹配首个连续出现)。
- **镜头切点**:每行一个时间点。

## 本地开发(可选)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```
