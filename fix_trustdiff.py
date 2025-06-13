#!/usr/bin/env python3
"""
TrustDiff 一键修复脚本
解决所有剩余的兼容性和配置问题
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description=""):
    """执行命令并处理错误"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} 完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 失败: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return False


def main():
    print("🚀 TrustDiff 一键修复工具")
    print("=" * 50)
    
    # 1. 检查当前目录
    if not Path("trustdiff").exists():
        print("❌ 请在 TrustDiff 项目根目录下运行此脚本")
        sys.exit(1)
    
    print("📍 检测到 TrustDiff 项目目录")
    
    # 2. 卸载旧版本
    print("\n🗑️  清理旧版本...")
    run_command("pip uninstall trustdiff -y", "卸载旧版本")
    
    # 3. 清理缓存文件
    print("\n🧹 清理缓存文件...")
    commands = [
        "find . -name '*.pyc' -delete",
        "find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
        "rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true"
    ]
    
    for cmd in commands:
        run_command(cmd, f"执行: {cmd}")
    
    # 4. 安装依赖
    print("\n📦 安装依赖...")
    run_command("pip install -r requirements.txt", "安装依赖包")
    
    # 5. 重新安装包
    print("\n🔄 重新安装 TrustDiff...")
    success = run_command("pip install --no-cache-dir -e .", "安装 TrustDiff")
    
    if not success:
        print("❌ 安装失败，请检查错误信息")
        sys.exit(1)
    
    # 6. 创建 .env 示例文件
    print("\n📝 创建配置文件...")
    env_example = """# TrustDiff API Keys
# 复制此文件为 .env 并填入你的 API 密钥

# OpenAI API Key (用于 baseline 和 judge)
OPENAI_API_KEY="sk-your-openai-key-here"

# 其他平台的 API Keys
TARGET_API_KEY="your-target-platform-key-here"
GEMINI_API_KEY="your-gemini-key-here"

# 可选：其他平台
# ANTHROPIC_API_KEY="your-anthropic-key"
# COHERE_API_KEY="your-cohere-key"
"""
    
    with open(".env.example", "w", encoding="utf-8") as f:
        f.write(env_example)
    
    print("✅ 创建了 .env.example 文件")
    
    # 7. 测试安装
    print("\n🧪 测试安装...")
    success = run_command("trustdiff --version", "测试 CLI 命令")
    
    if success:
        print("\n🎉 修复完成！")
        print("\n📋 后续步骤:")
        print("1. 复制 .env.example 为 .env: cp .env.example .env")
        print("2. 编辑 .env 文件，填入你的 API 密钥")
        print("3. 编辑 configs/default_config.yaml，配置你的平台")
        print("4. 运行评估: trustdiff --config configs/default_config.yaml")
        print("\n📚 更多信息请查看 QUICKSTART.md")
    else:
        print("❌ 安装测试失败，请检查错误信息")
        sys.exit(1)


if __name__ == "__main__":
    main() 