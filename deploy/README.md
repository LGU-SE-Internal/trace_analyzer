# Expdata — Experiment Data Service

集中存储和可视化 rLLM-SWE 实验数据的服务。实验脚本（eval/collection）跑完后自动上传结果到 K8s 上的 expdata 服务，通过 dashboard 可视化查看。

## Architecture

```
实验容器 (swe-eval-standalone.sh / swe-data-collection.sh)
    │  --upload --upload_url <EXPDATA_URL>
    ▼
Expdata Service (K8s Pod, FastAPI + SQLite)
    │  /api/v1/experiments, /api/v1/experiments/{id}/upload/*
    ▼
浏览器 (kubectl port-forward → http://localhost:8502)
```

- **认证**: 所有 API（除 `/healthz`、`/api/v1/auth/login`、dashboard 页面）都需要 Bearer token。首次连接自动用 admin/42 登录。
- **存储**: SQLite on PVC，数据持久化。
- **访问方式**:
  - 集群内部（实验容器）: `http://expdata.default.svc.cluster.local:8502`
  - 集群外部（开发机）: `kubectl port-forward svc/expdata 8502:8502` → `http://localhost:8502`

## Quick Start — 手动上传本地数据

```bash
# 上传单个实验目录
bash deploy/upload_local.sh data/expriment/Qwen3-8B

# 上传整个实验父目录（自动识别 eval/collection/rollout）
bash deploy/upload_local.sh data/expriment

# Dry run — 只预览不上传
bash deploy/upload_local.sh data/expriment --dry_run

# 指定服务器地址（默认走 port-forward localhost:8502）
bash deploy/upload_local.sh data/expriment --server http://other:8502
```

支持的目录格式：
- **Eval**: `<name>/*_n1.jsonl` + `<name>/chat_completions/eval.jsonl`
- **Collection**: `<name>/*.pos.jsonl` + `<name>/*.neg.jsonl`
- **Rollout**: `<name>/1.jsonl`, `<name>/2.jsonl`, ... (数字命名，每行为 messages list)

## Deploy to K8s

### Prerequisites

- `kubectl` 配置好目标集群
- `docker` 且能 push 到目标镜像仓库
- 集群有可用的 StorageClass（查看: `kubectl get storageclass`）

### Step 1: Configure

编辑 `deploy/expdata-k8s.yaml`：

```yaml
# PVC — 修改 storageClassName 为你集群的 StorageClass
storageClassName: ebs-ssd  # <- 改为你集群的 StorageClass
storage: 20Gi              # <- EBS 类存储最小 20Gi，按需调整

# Deployment — 修改 image 为你的镜像仓库地址
image: <your-registry>/<your-namespace>/expdata:latest
imagePullPolicy: Always
```

### Step 2: Build & Push Image

K8s 节点和本机架构可能不同（如 Apple Silicon vs amd64），需要 cross-build：

```bash
# 查看 K8s 节点架构
kubectl get nodes -o wide  # 看 OS-IMAGE / KERNEL 判断架构

# 从项目根目录构建并推送（以 amd64 为例）
docker buildx build --platform linux/amd64 \
    -f deploy/Dockerfile.expdata \
    -t <your-registry>/<your-namespace>/expdata:latest \
    --push .
```

如果 push 报 `unauthorized`：
```bash
docker login <your-registry>
# 输入凭据后重新 push
```

### Step 3: Deploy

```bash
kubectl apply -f deploy/expdata-k8s.yaml
```

### Step 4: Verify

```bash
# 检查资源状态
kubectl get pods -l app=expdata        # 应为 Running 1/1
kubectl get pvc expdata-pvc            # 应为 Bound

# 查看日志
kubectl logs -l app=expdata

# 从 pod 内部测试
kubectl exec $(kubectl get pods -l app=expdata -o jsonpath='{.items[0].metadata.name}') \
    -- python3 -c "
import urllib.request, json
data = json.dumps({'username':'admin','password':'42'}).encode()
req = urllib.request.Request('http://localhost:8502/api/v1/auth/login', data=data, headers={'Content-Type':'application/json'}, method='POST')
resp = urllib.request.urlopen(req)
print('Login:', json.loads(resp.read().decode()))
"
```

### Step 5: Access from Dev Machine

```bash
kubectl port-forward svc/expdata 8502:8502
# 浏览器打开 http://localhost:8502
```

Dashboard 页面需要输入 token。获取方式：
```bash
cat ~/.config/expdata/token
# 或首次使用: curl -s -X POST http://localhost:8502/api/v1/auth/login \
#   -H "Content-Type: application/json" -d '{"username":"admin","password":"42"}'
```

### Troubleshooting

| 症状 | 原因 | 解决 |
|------|------|------|
| Pod Pending + `unbound PVC` | StorageClass 不匹配 | `kubectl get storageclass` 确认名称，修改 yaml |
| Pod Pending + `InvalidVolumeSize` | 存储最小容量限制 | 增大 PVC storage（EBS 至少 20Gi） |
| ImagePullBackOff | 镜像不存在或没权限 | `docker push` + 确认 `imagePullPolicy: Always` |
| CrashLoopBackOff + health check 401 | probe 端点需要认证 | 确认 probe 路径为 `/healthz` |
| CrashLoopBackOff + `python-multipart` | Dockerfile 缺依赖 | 确认 Dockerfile 包含 `python-multipart` |
| Schema 不兼容 | 升级后旧 DB 结构冲突 | 删 PVC 重建: `kubectl delete pvc expdata-pvc && kubectl apply -f deploy/expdata-k8s.yaml` |
| Rollout upload 断连 | 单次上传数据量过大 | `import_local.py` 已内置分批上传（100条/批） |

### Update Image

```bash
docker buildx build --platform linux/amd64 \
    -f deploy/Dockerfile.expdata \
    -t <your-registry>/<your-namespace>/expdata:latest \
    --push .

kubectl rollout restart deployment expdata
```

## Use with Experiment Scripts

实验脚本默认自动上传。客户端首次连接自动用 admin/42 登录并缓存 token。

```bash
# Eval — 跑完自动上传
bash swe-eval-standalone.sh Qwen3-8B

# Collection — 跑完自动上传
bash swe-data-collection.sh

# 禁用上传
UPLOAD=false bash swe-eval-standalone.sh Qwen3-8B

# 自定义凭据
EXPDATA_USER=myname EXPDATA_PASSWORD=mypw bash swe-eval-standalone.sh Qwen3-8B
```

## Local Development

```bash
EXPDATA_DB_PATH=./data/expdata.db uvicorn utils.expdata.server:app --port 8502
# 浏览器打开 http://localhost:8502
```
