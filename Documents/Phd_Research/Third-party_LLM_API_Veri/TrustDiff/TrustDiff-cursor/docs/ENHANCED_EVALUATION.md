# Enhanced LLM-as-a-Judge Evaluation

TrustDiff现在支持基于README.md规范的增强型LLM-as-a-Judge评估系统，提供详细的多维度质量评分。

## 核心功能

### 1. 详细评分维度

每个回答在以下4个维度上获得1-5分的评分：

- **Correctness (正确性)**: 事实和逻辑准确性
- **Reasoning_Depth (推理深度)**: 推理过程的深度和解释质量
- **Instruction_Adherence (指令遵循)**: 对明确和隐含指令的遵循程度
- **Clarity_Conciseness (清晰简洁)**: 回答的清晰度、结构和简洁性

### 2. 结构化输出格式

Judge LLM返回详细的JSON结构：

```json
{
  "scores": {
    "answer_a": {
      "correctness": 4,
      "reasoning_depth": 3,
      "instruction_adherence": 5,
      "clarity_conciseness": 4
    },
    "answer_b": {
      "correctness": 5,
      "reasoning_depth": 5,
      "instruction_adherence": 5,
      "clarity_conciseness": 5
    }
  },
  "comparative_reasoning": "Answer B demonstrates superior reasoning depth with clear step-by-step logic...",
  "final_verdict": "BASELINE_SUPERIOR"
}
```

### 3. 量化质量分析

报告自动计算：

- 各维度的平均得分差异
- 质量下降百分比
- 最受影响的维度识别
- 整体质量趋势分析

## 使用方法

### 1. 配置Judge LLM

确保在配置文件中正确设置judge配置：

```yaml
judge:
  name: "Judge_LLM"
  api_base: "https://api.openai.com/v1"
  api_key_env: "judge_llm_key"
  model: "gpt-4"  # 推荐使用GPT-4以获得最佳评估质量
```

### 2. 运行增强评估

```bash
# 运行测试，自动使用增强评估
trustdiff run --config configs/default_config.yaml

# 生成详细报告
trustdiff report outputs/2024-01-01_10-30-00/
```

### 3. 查看详细分析

生成的报告将包含：

#### 详细质量分析表格
| Dimension | Target | Baseline | Difference | Degradation % |
|-----------|--------|----------|------------|---------------|
| Correctness | 3.85 | 4.20 | -0.35 | 8.3% |
| Reasoning Depth | 3.10 | 4.80 | -1.70 | 35.4% |
| Instruction Adherence | 4.15 | 4.30 | -0.15 | 3.5% |
| Clarity Conciseness | 3.95 | 4.25 | -0.30 | 7.1% |

#### 关键发现
- **最受影响的维度**: Reasoning Depth显示35.4%的平均下降
- **整体质量影响**: 所有维度平均13.6%的下降

## 数据存储

详细评分数据存储在SQLite数据库中，包含：

- 原始verdict和reasoning
- 每个维度的具体分数
- 比较性推理过程
- 置信度评分

## 高级用法

### 自定义Judge Prompt

可以通过修改`comparator.py`中的`judge_prompt_template`来自定义评估标准。

### 编程式访问

```python
from trustdiff import Engine, QualityEvaluation, DetailedScores

# 访问详细评分
evaluation = quality_evaluation
if evaluation.detailed_scores_target:
    target_scores = evaluation.detailed_scores_target
    print(f"Target Correctness: {target_scores.correctness}")
    print(f"Target Reasoning: {target_scores.reasoning_depth}")
```

### 批量分析

```python
# 分析特定维度的表现
import pandas as pd

# 从数据库加载结果
evaluations_df = pd.read_sql_query("""
    SELECT probe_id, target_platform,
           target_correctness, baseline_correctness,
           target_reasoning_depth, baseline_reasoning_depth
    FROM evaluations 
    WHERE evaluation_success = 1
""", conn)

# 计算推理深度平均下降
reasoning_degradation = evaluations_df.groupby('target_platform').apply(
    lambda x: ((x['baseline_reasoning_depth'].mean() - x['target_reasoning_depth'].mean()) 
               / x['baseline_reasoning_depth'].mean() * 100)
)
```

## 最佳实践

1. **使用GPT-4作为Judge**: 获得最一致和准确的评估
2. **设计多样化探针**: 测试不同类型的推理和任务
3. **分析趋势而非单个结果**: 关注统计显著的模式
4. **验证评估质量**: 定期检查judge的评估是否合理
5. **结合多种指标**: 不仅依赖质量评分，还要考虑延迟和成本

## 故障排除

### 常见问题

1. **Judge返回非JSON格式**: 检查API key和模型配置
2. **评分解析失败**: 确保judge model支持复杂instruction following
3. **评估结果不一致**: 考虑降低temperature或使用更稳定的模型

### 调试技巧

启用详细日志查看judge的原始响应：

```bash
export TRUSTDIFF_DEBUG=1
trustdiff run --config configs/default_config.yaml
```

这个增强的评估系统为您的研究提供了科学、严谨、量化的质量度量工具，能够以不可辩驳的方式证明和度量LLM API平台之间的质量差异。 