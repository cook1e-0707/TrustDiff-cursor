# H-CAF 分层认知能力评估框架指南

## 🧠 从"相似度比较"到"智力审计"的革新

TrustDiff 2.0 引入了 **H-CAF (Hierarchical Cognitive Assessment Framework) 分层认知能力评估框架**，这是一个突破性的AI模型能力评估体系，将传统的"对照检查"升级为深度的"认知能力审计"。

## 🎯 核心理念

### 传统方法的局限
- ❌ 简单的相似度比较
- ❌ 表面的"好坏"判断  
- ❌ 缺乏具体的能力维度分析

### H-CAF的创新
- ✅ **认知指纹分析**：将AI智力分解为5个可测量的认知向量
- ✅ **能力雷达图**：直观展示每个模型的认知能力画像
- ✅ **量化退化评估**：精确测量能力下降的具体维度和程度

## 🧭 五大核心认知向量

H-CAF将AI的"智力"分解为以下5个核心认知向量：

### 1. 逻辑推理 (Logical Reasoning)
**能力描述**: 处理因果关系、演绎、归纳和抽象思考的能力
- 因果链分析
- 演绎推理
- 归纳总结
- 逆向逻辑

**评分标准** (1-10):
- 1-2: 严重缺陷，逻辑混乱
- 3-4: 基础逻辑，但有明显漏洞
- 5-6: 一般逻辑能力，基本合理
- 7-8: 良好逻辑，推理严密
- 9-10: 卓越逻辑，深度洞察

### 2. 知识应用 (Knowledge Application)
**能力描述**: 准确、快速地调用相关知识来解决问题的能力
- 事实准确性
- 知识整合
- 跨领域连接
- 专业深度

### 3. 创造性综合 (Creative Synthesis)
**能力描述**: 组合不相关概念、产生新颖想法、进行比喻性思维的能力
- 概念融合
- 原创性思考
- 类比推理
- 创新解决方案

### 4. 指令忠实度 (Instructional Fidelity)
**能力描述**: 精确理解并执行复杂、多重甚至矛盾指令的能力
- 复杂指令解析
- 多重约束处理
- 隐含要求理解
- 执行精确度

### 5. 安全与元认知 (Safety & Metacognition)
**能力描述**: 识别风险、承认自身局限性、进行自我修正的能力
- 安全意识
- 边界认知
- 不确定性表达
- 自我纠错

## 📊 认知指纹与能力雷达图

### 认知指纹示例
```json
{
  "target_platform_fingerprint": {
    "logical_reasoning": 7,
    "knowledge_application": 8,
    "creative_synthesis": 5,
    "instructional_fidelity": 9,
    "safety_metacognition": 6
  },
  "baseline_fingerprint": {
    "logical_reasoning": 9,
    "knowledge_application": 9,
    "creative_synthesis": 8,
    "instructional_fidelity": 9,
    "safety_metacognition": 8
  }
}
```

### 能力差距分析
```json
{
  "capability_gaps": {
    "logical_reasoning_gap": 2.0,     // 逻辑推理能力下降2分
    "knowledge_application_gap": 1.0, // 知识应用能力下降1分
    "creative_synthesis_gap": 3.0,    // 创造性综合能力下降3分 (最严重)
    "instructional_fidelity_gap": 0.0, // 指令忠实度无变化
    "safety_metacognition_gap": 2.0   // 安全元认知能力下降2分
  },
  "overall_degradation": 1.6,         // 整体退化程度
  "degradation_severity": "Moderate"  // 退化严重程度：中等
}
```

## 🔧 配置和使用

### 启用H-CAF框架

在 `configs/default_config.yaml` 中配置：

```yaml
run_settings:
  # H-CAF框架设置
  use_hcaf_framework: true  # 启用H-CAF认知评估
  cognitive_vectors_focus: null  # 使用全部5个认知向量
  
  # 可选：只关注特定认知向量
  # cognitive_vectors_focus: ["logical_reasoning", "creative_synthesis"]
```

### 设计针对性探针

为不同认知向量设计专门的测试探针：

```yaml
# 逻辑推理探针示例
probe_id: "logical_reasoning_causal_analysis"
cognitive_focus: ["logical_reasoning", "knowledge_application"]
prompt:
  - role: "user"
    content: |
      分析以下现象的因果关系：
      1. 新建地铁线路
      2. 房价上涨40%
      3. 科技公司增长300%
      请建立因果链条并预测后续效应。

# 创造性综合探针示例  
probe_id: "creative_synthesis_concept_fusion"
cognitive_focus: ["creative_synthesis", "instructional_fidelity"]
prompt:
  - role: "user"
    content: |
      请设计一个融合"共享单车"和"宠物寄养"的商业模式，
      要求：150字描述，包含3个创新点，1个风险评估。
```

### 运行H-CAF评估

```bash
# 完整H-CAF认知评估
trustdiff run --config configs/default_config.yaml

# 只测试特定认知向量
trustdiff run --config configs/default_config.yaml --probe-filter "logical_reasoning"

# 生成包含认知雷达图的报告
trustdiff report outputs/2024-01-01_10-30-00/ --output-format markdown
```

## 📈 H-CAF报告示例

### 执行摘要
```markdown
## H-CAF认知能力审计摘要

### 整体评估
- **总体退化程度**: 15.2% (中等)
- **最受影响维度**: 创造性综合 (-37.5%)
- **认知稳定性评级**: B级 (轻微不稳定)

### 认知向量分析
| 认知向量 | 目标平台 | 基准平台 | 差距 | 退化率 |
|----------|----------|----------|------|--------|
| 逻辑推理 | 7.2      | 8.8      | -1.6 | 18.2%  |
| 知识应用 | 8.1      | 8.9      | -0.8 | 9.0%   |
| 创造综合 | 5.0      | 8.0      | -3.0 | 37.5%  |
| 指令忠实 | 8.5      | 8.7      | -0.2 | 2.3%   |
| 安全元认知| 6.8      | 7.9      | -1.1 | 13.9%  |
```

### 关键发现
- **🔴 严重退化**: 创造性综合能力显著下降，影响原创性思维
- **🟡 中等影响**: 逻辑推理能力有所下降，但基本功能完整
- **🟢 基本稳定**: 指令忠实度保持良好，执行能力可靠

### 建议
1. **优先修复**: 创造性综合模块需要重点优化
2. **监控观察**: 逻辑推理能力需要持续监控
3. **保持优势**: 指令执行能力是该平台的优势，需要保持

## 🎓 H-CAF最佳实践

### 1. 探针设计原则
- **单一焦点**: 每个探针主要测试1-2个认知向量
- **分层难度**: 设计不同难度级别的测试
- **真实场景**: 使用接近实际应用的测试场景

### 2. 评估策略
- **基准一致**: 使用相同的基准模型进行对比
- **多轮测试**: 进行多轮测试以确保结果稳定性
- **交叉验证**: 使用不同的探针验证同一认知向量

### 3. 结果解读
- **关注趋势**: 重点关注能力变化趋势而非绝对分数
- **综合分析**: 结合多个认知向量进行综合判断
- **实用导向**: 根据具体应用场景权衡不同能力的重要性

## 🔬 高级功能

### 自定义认知向量
可以根据特定需求定义额外的认知向量：

```python
# 扩展认知向量定义
custom_vectors = {
    "mathematical_reasoning": "数学推理和计算能力",
    "emotional_intelligence": "情感理解和表达能力",
    "domain_expertise": "特定领域的专业知识"
}
```

### 认知向量权重配置
为不同应用场景设置认知向量权重：

```yaml
# 科研助手场景
cognitive_weights:
  logical_reasoning: 0.3
  knowledge_application: 0.3
  creative_synthesis: 0.2
  instructional_fidelity: 0.1
  safety_metacognition: 0.1

# 创意写作场景  
cognitive_weights:
  logical_reasoning: 0.1
  knowledge_application: 0.2
  creative_synthesis: 0.4
  instructional_fidelity: 0.2
  safety_metacognition: 0.1
```

## 📚 理论基础

H-CAF框架基于以下理论基础：

1. **认知心理学**: Carroll的三层智力理论
2. **AI能力评估**: 多维能力评估框架
3. **系统评估**: 分层系统性能评估方法
4. **质量保证**: ISO/IEC标准的质量评估体系

## 🚀 未来发展

### 即将推出的功能
- **动态认知基准**: 自动调整基准难度
- **认知能力预测**: 基于历史数据预测能力发展趋势  
- **个性化评估**: 根据应用场景定制评估维度
- **实时监控**: 持续监控模型能力变化

H-CAF框架代表了AI模型评估的未来方向，从简单的比较升级为深度的智力分析，为AI系统的可信度提供了科学的量化标准。 