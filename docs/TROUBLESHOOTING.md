# TrustDiff 故障排除指南

## 🚨 常见问题及解决方案

### 1. API调用失败但显示Success

**症状**: 日志显示 HTTP 404 或其他错误，但最终报告显示 100% success rate

**原因**: 错误处理逻辑问题

**解决方案**:
- 确保使用最新版本的TrustDiff
- 检查API端点URL是否正确
- 使用 `trustdiff debug` 命令验证配置

```bash
# 验证配置
trustdiff debug --config configs/default_config.yaml

# 查看详细的错误信息
trustdiff run --config configs/default_config.yaml --probe-filter "reasoning_bucket"
```

### 2. URL拼接错误 (双斜杠)

**症状**: 错误信息显示 `/v1//chat/completions` (双斜杠)

**原因**: API base URL配置问题

**解决方案**:
```yaml
# ❌ 错误配置
api_base: "https://api.example.com/v1/"

# ✅ 正确配置
api_base: "https://api.example.com/v1"
```

### 3. 第三方平台认证失败

**症状**: HTTP 401, 403 或认证相关错误

**可能原因及解决方案**:

#### A. API密钥未设置或错误
```bash
# 检查环境变量
echo $PLATFORM_A_KEY

# 重新设置环境变量
export PLATFORM_A_KEY="your-actual-api-key"
```

#### B. 认证格式不匹配
有些第三方平台可能需要不同的认证格式：

```yaml
# 标准OpenAI格式 (默认)
- name: "Standard_Platform"
  api_base: "https://api.platform.com/v1"
  api_key_env: "platform_key"
  auth_type: "bearer"
  api_key_prefix: "Bearer"

# 自定义认证格式
- name: "Custom_Platform"
  api_base: "https://api.custom.com/v1"
  api_key_env: "custom_key"
  auth_type: "api_key"
  api_key_header: "X-API-Key"

# 完全自定义
- name: "Special_Platform"
  api_base: "https://api.special.com"
  api_key_env: "special_key"
  auth_type: "custom"
  headers:
    "X-API-Key": "${SPECIAL_KEY}"
    "X-Client-ID": "trustdiff"
```

### 4. 网络连接问题

**症状**: Connection timeout, DNS resolution failed

**解决方案**:
1. 检查网络连接
2. 验证URL是否可访问
3. 检查防火墙设置
4. 增加超时时间：

```yaml
run_settings:
  timeout_seconds: 60  # 增加到60秒
  concurrency: 5       # 降低并发数
```

### 5. Gemini API调用失败

**症状**: Gemini裁判模型调用失败

**解决方案**:

#### A. API密钥配置
```bash
# 获取Gemini API Key
# 访问: https://aistudio.google.com/app/apikey

# 设置环境变量
export GEMINI_API_KEY="AIzaSy..."
```

#### B. 网络访问问题
```yaml
# 如果无法访问Google API，可以改用OpenAI作为裁判
judge:
  name: "GPT4_Judge"
  api_base: "https://api.openai.com/v1"
  api_key_env: "openai"
  model: "gpt-4o"
```

### 6. 模型不支持错误

**症状**: "model not found" 或类似错误

**解决方案**:
1. 检查平台支持的模型列表
2. 更新配置文件中的模型名称：

```yaml
# 常见的模型名称
targets:
  - name: "Platform_A"
    model: "gpt-4o"           # OpenAI标准
  - name: "Platform_B"  
    model: "gpt-4-turbo"      # 某些平台的命名
  - name: "Platform_C"
    model: "chatgpt-4"        # 其他可能的命名
```

### 7. 速率限制问题

**症状**: HTTP 429 "Too Many Requests"

**解决方案**:
```yaml
run_settings:
  concurrency: 1        # 降低并发数
  timeout_seconds: 45   # 增加超时时间

# 或者在代码中添加延迟（高级用法）
```

## 🔧 调试命令

### 基本调试
```bash
# 验证配置
trustdiff debug --config configs/default_config.yaml

# 测试单个探针
trustdiff run --config configs/default_config.yaml --probe-filter "bucket_problem"

# 预览运行（不实际执行）
trustdiff run --config configs/default_config.yaml --dry-run
```

### 详细日志
```bash
# 启用详细调试信息
export TRUSTDIFF_DEBUG=1
trustdiff run --config configs/default_config.yaml
```

## 🔍 配置验证清单

运行测试前，请检查以下项目：

- [ ] 所有API密钥已正确设置
- [ ] API端点URL格式正确（无尾随斜杠）
- [ ] 网络可以访问所有配置的端点
- [ ] 模型名称与平台支持的模型匹配
- [ ] 认证格式正确
- [ ] 并发数和超时时间适合你的网络环境

## 📞 获取帮助

如果问题仍然存在：

1. **运行调试命令**:
   ```bash
   trustdiff debug --config configs/default_config.yaml
   ```

2. **检查详细日志**:
   查看 `outputs/[timestamp]/logs/` 目录下的JSON文件

3. **验证配置**:
   确认 `configs/default_config.yaml` 中的所有设置

4. **简化测试**:
   先只配置一个目标平台进行测试

5. **检查API文档**:
   确认第三方平台的API格式和认证方式

## 🛠️ 高级故障排除

### 自定义请求格式
对于非标准OpenAI格式的API：

```python
# 在 engine.py 中可以添加自定义请求处理
# 这需要修改源代码，适合高级用户
```

### 日志分析
```bash
# 查看原始API响应
cat outputs/2024-01-01_10-30-00/logs/*.json | jq .

# 分析错误模式
grep -r "error" outputs/2024-01-01_10-30-00/logs/
```

记住：大多数问题都是配置相关的，仔细检查API密钥、URL和认证设置通常可以解决问题。 