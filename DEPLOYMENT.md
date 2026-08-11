# Anima-MLX 本地 API 部署

本部署使用 `xocialize/anima-mlx` 的 int4 主干，在 Apple Silicon 上提供 OpenAI Images 兼容接口。
模型权重受 CircleStone 非商业许可证约束，只可用于个人或研究用途；生成图片本身可按原模型许可使用。

## 服务地址

- API：`http://Mac.lan:10240/v1`（按实际局域网主机名或 IP 修改）
- 健康检查：`GET /health`
- 文生图：`POST /v1/images/generations`
- 默认：512×512、30 步、CFG 5、每次一张、串行生成

API 密钥保存在本机 `.env` 和调用方配置中，不应提交到 Git。

## 安装与启动

```bash
uv venv .venv --python 3.12
uv sync
cp .env.example .env
```

编辑 `.env` 中的 API 密钥、输出目录和公开访问地址，然后启动：

```bash
scripts/run_api.sh
```

本机接口测试：

```bash
cd /absolute/path/to/anima-mlx
set -a; source .env; set +a
curl -sS http://127.0.0.1:10240/v1/images/generations \
  -H "Authorization: Bearer $ANIMA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"anima-mlx","prompt":"masterpiece, best quality, safe, furina (genshin impact), delighted smile, theatrical pose","size":"512x512","steps":30}'
```

## AstrBot 使用

- 插件项目：[`tignioj/astrbot_plugin_anima_furina`](https://github.com/tignioj/astrbot_plugin_anima_furina)
- 手动测试：`/芙宁娜表情 开心 在舞台谢幕`
- 自动调用工具：`furina_emotion_image`

工具说明已经要求大模型仅在图片能明显增强角色扮演的情绪表达时调用，避免每轮对话都生图。

## macOS 常驻运行

复制并编辑 `deploy/com.local.anima-mlx-api.plist.example`，将其中的
`/ABSOLUTE/PATH/TO/anima-mlx` 替换为项目绝对路径，然后保存为：

```text
~/Library/LaunchAgents/com.local.anima-mlx-api.plist
```

加载与管理：

```bash
launchctl kickstart -k gui/$(id -u)/com.local.anima-mlx-api
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.local.anima-mlx-api.plist
curl http://127.0.0.1:10240/health -H "Authorization: Bearer $ANIMA_API_KEY"
```

日志位于项目的 `logs/api.stdout.log` 和 `logs/api.stderr.log`。

Mac 进入系统睡眠后局域网服务会暂停；如需机器人全天可用，应在电源设置中让 Mac 接电时保持唤醒。合盖睡眠仍会中断服务。
