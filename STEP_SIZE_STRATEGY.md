# Step Size Selection Strategies

## Overview
实现了两种步长选择策略用于全局布局优化：

### 1. Barzilai-Borwein (BB) Method
- **原理**: 基于梯度变化的自适应步长
- **公式**: `α = |Δg · Δx| / ||Δg||²`
- **优点**: 计算简单，收敛稳定
- **适用**: 小中规模问题

### 2. ePlace Backtracking
- **原理**: 带回溯的线搜索
- **参数**: 
  - `backtrack_epsilon`: 收缩因子 (默认0.95)
  - `max_backtrack`: 最大回溯次数 (默认10)
- **优点**: 保证目标函数下降
- **适用**: 大规模复杂问题

## Usage

### 配置文件 (JSON)

```json
{
  "step_size_strategy": "bb",  // 或 "eplace"
  "backtrack_epsilon": 0.95,
  "max_backtrack": 10,
  "global_place_stages": [
    {
      "optimizer": "nesterov",
      "step_size_strategy": "bb"  // 可覆盖全局设置
    }
  ]
}


