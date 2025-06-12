# TrustDiff 配置指南

## 📍 配置文件位置

主要配置都在 `configs/default_config.yaml` 文件中进行。

## 🔑 API密钥配置

### 1. 设置环境变量

在命令行中设置以下环境变量：

#### Windows (PowerShell):
```powershell
$env:OPENAI_API_KEY="sk-your-openai-api-key-here"
$env:PLATFORM_A_KEY="your-platform-a-api-key-here"
$env:GEMINI_API_KEY="your-gemini-api-key-here"
```

#### Linux/macOS (Bash):
```bash
export OPENAI_API_KEY="sk-your-openai-api-key-here"
export PLATFORM_A_KEY="your-platform-a-api-key-here"
export GEMINI_API_KEY="your-gemini-api-key-here"
```

#### 或创建 `.env` 文件:
在项目根目录创建 `.env` 文件：
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
PLATFORM_A_KEY=your-platform-a-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
```

### 2. API密钥获取方式

#### OpenAI API Key
- 访问：https://platform.openai.com/api-keys
- 创建新的API密钥
- 格式：`sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

#### Gemini API Key
- 访问：https://makersuite.google.com/app/apikey
- 或者：https://aistudio.google.com/app/apikey
- 创建新的API密钥
- 格式：`AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX`

#### 第三方平台API Key
- 根据具体平台的文档获取
- 通常格式与OpenAI类似

## 🎯 被检测平台配置

在 `configs/default_config.yaml` 的 `targets` 部分配置待检测平台：

### 示例配置：

```yaml
targets:
  # 第三方OpenAI API代理平台
  - name: "SomeCloudAPI"
    api_base: "https://api.somecloud.com/v1"    # 替换为实际URL
    api_key_env: "platform_a_key"               # 对应环境变量名
    model: "gpt-4o"
  
  # 自建代理（如NewAPI/One-API）
  - name: "Self_Hosted_Proxy" 
    api_base: "http://localhost:3000/v1"        # 本地代理地址
    api_key_env: "openai"                       # 可复用OpenAI密钥
    model: "gpt-4o"
  
  # 其他平台示例
  - name: "Platform_B"
    api_base: "https://api.platform-b.com/v1"
    api_key_env: "platform_b_key"
    model: "gpt-4o"
```

### 配置说明：

- **name**: 平台显示名称（用于报告中识别）
- **api_base**: API基础URL（必须与OpenAI兼容）
- **api_key_env**: 对应的环境变量名
- **model**: 使用的模型名称

## 👨‍⚖️ 裁判模型配置

### 使用Gemini作为裁判：

```yaml
judge:
  name: "Gemini_Judge"
  api_base: "https://generativelanguage.googleapis.com/v1beta"
  api_key_env: "gemini_key"
  model: "gemini-1.5-pro"  # 可选择其他Gemini模型
```

### 可用的Gemini模型：
- `gemini-1.5-pro`
- `gemini-1.5-flash`
- `gemini-2.0-flash-exp`

### 使用OpenAI GPT-4作为裁判：

```yaml
judge:
  name: "GPT4_Judge"
  api_base: "https://api.openai.com/v1"
  api_key_env: "openai"
  model: "gpt-4o"
```

## 🏃‍♂️ 基准平台配置

建议使用官方OpenAI作为基准：

```yaml
baseline:
  name: "OpenAI_Official"
  api_base: "https://api.openai.com/v1"
  api_key_env: "openai"
  model: "gpt-4o"  # 使用强模型获得更好的基准质量
```

## 🔧 运行参数配置

```yaml
run_settings:
  probe_dir: "./probes"           # 探针文件目录
  output_dir: "./outputs"         # 结果输出目录
  concurrency: 10                 # 并发请求数（根据API限制调整）
  timeout_seconds: 30             # 请求超时时间
```

## ✅ 配置验证

### 1. 测试配置
```bash
# 验证配置文件
trustdiff run --config configs/default_config.yaml --dry-run
```

### 2. 检查API连接
```bash
# 测试单个探针
trustdiff run --config configs/default_config.yaml --probe-filter "reasoning_bucket"
```

## 🚨 常见问题

### 1. API密钥错误
- 检查环境变量是否正确设置
- 验证API密钥格式
- 确认密钥有效且有余额

### 2. 网络连接问题
- 检查API端点URL是否正确
- 验证网络连接和防火墙设置
- 增加 `timeout_seconds` 值

### 3. 模型不支持
- 确认目标平台支持指定的模型
- 检查模型名称拼写
- 查看平台文档了解支持的模型列表

### 4. Gemini API特殊设置
- 确保使用正确的API端点
- 检查API密钥权限
- 某些地区可能需要VPN

## 📝 示例完整配置

```yaml
# API Keys
api_keys:
  openai: ${OPENAI_API_KEY}
  some_platform_key: ${SOME_PLATFORM_KEY}
  gemini_key: ${GEMINI_API_KEY}

# 基准平台
baseline:
  name: "OpenAI_Official"
  api_base: "https://api.openai.com/v1"
  api_key_env: "openai"
  model: "gpt-4o"

# 待检测平台
targets:
  - name: "SomeCloudAPI"
    api_base: "https://api.somecloud.com/v1"
    api_key_env: "some_platform_key"
    model: "gpt-4o"

# 裁判模型
judge:
  name: "Gemini_Judge"
  api_base: "https://generativelanguage.googleapis.com/v1beta"
  api_key_env: "gemini_key"
  model: "gemini-1.5-pro"

# 运行设置
run_settings:
  probe_dir: "./probes"
  output_dir: "./outputs"
  concurrency: 5      # 保守的并发数
  timeout_seconds: 45 # 较长的超时时间
```

## 🎯 快速开始

1. **复制示例配置**：已在 `configs/default_config.yaml` 中
2. **设置环境变量**：设置所需的API密钥
3. **修改配置**：更新平台URL和模型设置
4. **测试运行**：`trustdiff run --config configs/default_config.yaml --dry-run`
5. **正式运行**：`trustdiff run --config configs/default_config.yaml`

配置完成后，您就可以开始检测LLM API平台的"语义漏洞"了！ 