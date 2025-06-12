# 🔧 TrustDiff 快速修复指南

## 问题1: Gemini API响应格式错误

### 症状
```
Unexpected Gemini response format: {'candidates': [{'content': {'role': 'model'}, 'finishReason': 'MAX_TOKENS', 'index': 0}]}
```

### 解决方案
已在新版本中修复Gemini API响应解析逻辑，支持多种响应格式和错误处理。

### 测试命令
```bash
# 测试Gemini API连接
trustdiff test-gemini

# 测试所有可用的Gemini模型
trustdiff test-gemini --test-models

# 测试特定模型
trustdiff test-gemini --model "gemini-1.5-pro"
```

## 问题2: 大量负数延迟差异

### 症状
报告中显示大量负数延迟差异（如-17896.9ms）

### 说明
- **负数延迟差异是正常的**：表示目标平台比基准平台更快
- **大数值可能的原因**：
  - 网络波动
  - 平台负载差异
  - API响应时间变化

### 解释
- `latency_diff_ms = target_latency - baseline_latency`
- 负数 = 目标平台更快
- 正数 = 目标平台更慢

### 新的报告格式
现在报告会显示更易读的格式：
```
| target_platform | latency_vs_baseline | quality_verdict |
|------------------|--------------------|-----------------| 
| Platform_A       | 150.2ms (slower)   | BASELINE_SUPERIOR |
| Platform_B       | 50.3ms (faster)    | SIMILAR_QUALITY   |
```

## 问题3: Gemini API配置问题

### 推荐配置
```yaml
judge:
  name: "Gemini_Judge"
  api_base: "https://generativelanguage.googleapis.com/v1beta"
  api_key_env: "gemini_key"
  model: "gemini-1.5-pro"    # 推荐稳定版本
```

### 可选模型
- `gemini-1.5-pro` (推荐，稳定)
- `gemini-1.5-flash` (更快，成本更低)
- `gemini-2.0-flash-exp` (实验版本)

### 环境变量设置
```bash
# 获取API密钥: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="AIzaSy..."
```

## 快速诊断命令

### 1. 完整配置诊断
```bash
trustdiff debug --config configs/default_config.yaml
```

### 2. Gemini专项测试
```bash
# 基础测试
trustdiff test-gemini

# 全面测试
trustdiff test-gemini --test-models --model "gemini-1.5-pro"
```

### 3. 单个探针测试
```bash
trustdiff run --config configs/default_config.yaml --probe-filter "reasoning_bucket"
```

## 验证修复

运行以下命令验证修复是否成功：

```bash
# 1. 验证基础配置
python test_fixes.py

# 2. 测试Gemini API
trustdiff test-gemini

# 3. 运行简单测试
trustdiff run --config configs/default_config.yaml --probe-filter "reasoning" --dry-run

# 4. 完整测试运行
trustdiff run --config configs/default_config.yaml
```

## 预期结果

修复后应该看到：
- ✅ 不再有"Unexpected Gemini response format"错误
- ✅ 延迟差异有清晰的faster/slower标识
- ✅ Gemini API正常工作并返回JSON格式的评估
- ✅ 报告显示准确的成功率统计

## 如果问题仍然存在

1. **检查API密钥**:
   ```bash
   echo $GEMINI_API_KEY
   echo $OPENAI_API_KEY
   echo $PLATFORM_A_KEY
   ```

2. **检查网络连接**:
   ```bash
   curl -H "Content-Type: application/json" \
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=$GEMINI_API_KEY" \
        -d '{"contents":[{"parts":[{"text":"Hello"}]}]}'
   ```

3. **降低并发和增加超时**:
   ```yaml
   run_settings:
     concurrency: 1
     timeout_seconds: 60
   ```

4. **使用OpenAI作为替代裁判**:
   ```yaml
   judge:
     name: "GPT4_Judge"
     api_base: "https://api.openai.com/v1"
     api_key_env: "openai"
     model: "gpt-4o"
   ```

记住：这些修复解决了API调用、响应解析和报告格式的所有已知问题！ 