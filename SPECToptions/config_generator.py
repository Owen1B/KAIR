#!/usr/bin/env python3
"""
配置文件生成器
从基础模板和实验配置生成最终的JSON配置文件
"""

import json
import os
import re
from typing import Dict, Any
import argparse

def remove_json_comments(text):
    """移除JSON文件中的注释"""
    # 移除 // 注释
    text = re.sub(r'//.*?(?=\n|$)', '', text)
    # 移除多余的逗号
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)
    return text

def load_template(template_path: str) -> str:
    """加载基础模板"""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def load_network_configs(network_config_path: str) -> Dict[str, Any]:
    """加载网络配置"""
    with open(network_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 保留注释用于最终输出
        clean_content = remove_json_comments(content)
        return json.loads(clean_content)

def load_experiment_configs(experiment_config_path: str) -> Dict[str, Any]:
    """加载实验配置"""
    with open(experiment_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        clean_content = remove_json_comments(content)
        return json.loads(clean_content)

def load_network_config_with_comments(network_config_path: str, network_type: str) -> str:
    """加载网络配置（保留注释）"""
    with open(network_config_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # 解析JSON以找到对应的网络配置
        clean_content = remove_json_comments(content)
        configs = json.loads(clean_content)
        
        # 从原始内容中提取对应网络的配置（保留注释）
        lines = content.split('\n')
        in_target_config = False
        config_lines = []
        brace_count = 0
        
        for line in lines:
            if f'"{network_type}":' in line:
                in_target_config = True
                continue
            elif in_target_config:
                if '{' in line:
                    brace_count += line.count('{')
                if '}' in line:
                    brace_count -= line.count('}')
                    if brace_count <= 0:
                        config_lines.append(line.replace('}', '').rstrip())
                        break
                config_lines.append(line)
        
                 # 构建最终的网络配置字符串
        result = "{\n"
        # 找到最后一个非空行的索引
        last_non_empty_index = -1
        for i in range(len(config_lines) - 1, -1, -1):
            if config_lines[i].strip():
                last_non_empty_index = i
                break
        
        for i, line in enumerate(config_lines):
            if line.strip():
                clean_line = line.strip()
                # 如果是最后一个非空行且以逗号结尾，移除逗号
                if i == last_non_empty_index and clean_line.endswith(','):
                    clean_line = clean_line[:-1]
                result += "    " + clean_line + "\n"
        result += "}"
        
        return result

def generate_config(template: str, experiment_config: Dict[str, Any], 
                   network_configs: Dict[str, Any], network_config_path: str) -> str:
    """生成最终配置"""
    config = template
    
    # 替换网络配置（保留注释）
    network_type = experiment_config['NETWORK_CONFIG']
    network_config_with_comments = load_network_config_with_comments(network_config_path, network_type)
    config = config.replace('"{{NETWORK_CONFIG}}"', network_config_with_comments)
    
    # 替换其他变量
    for key, value in experiment_config.items():
        if key != 'NETWORK_CONFIG':
            # 处理特殊的数据类型
            if isinstance(value, bool):
                value_str = str(value).lower()
            elif value is None:
                value_str = "null"
            elif isinstance(value, str):
                value_str = f'"{value}"'
            elif isinstance(value, float):
                # 保持科学记数法格式
                if value == 1e-4:
                    value_str = "1e-4"
                elif value == 2.5e-5:
                    value_str = "2.5e-5"
                elif value == 1e-9:
                    value_str = "1e-9"
                elif value == 1e-6:
                    value_str = "1e-6"
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
            
            placeholder = f'"{{{{{key}}}}}"'
            config = config.replace(placeholder, value_str)
    
    return config

def main():
    parser = argparse.ArgumentParser(description='生成配置文件')
    parser.add_argument('--experiment', '-e', type=str, required=True,
                       help='实验名称')
    parser.add_argument('--output-dir', '-o', type=str, 
                       default='SPECToptions/8x/generated',
                       help='输出目录')
    parser.add_argument('--template', '-t', type=str, 
                       default='SPECToptions/base_template.json',
                       help='模板文件路径')
    parser.add_argument('--network-configs', '-n', type=str,
                       default='SPECToptions/network_configs.json',
                       help='网络配置文件路径')
    parser.add_argument('--experiment-configs', '-c', type=str,
                       default='SPECToptions/experiment_configs.json',
                       help='实验配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    template = load_template(args.template)
    network_configs = load_network_configs(args.network_configs)
    experiment_configs = load_experiment_configs(args.experiment_configs)
    
    # 检查实验是否存在
    if args.experiment not in experiment_configs['experiments']:
        print(f"错误：实验 '{args.experiment}' 不存在")
        print("可用的实验：")
        for exp_name in experiment_configs['experiments'].keys():
            print(f"  - {exp_name}")
        return
    
    # 生成配置
    experiment_config = experiment_configs['experiments'][args.experiment]
    final_config = generate_config(template, experiment_config, network_configs, args.network_configs)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存文件
    output_file = os.path.join(args.output_dir, f"{args.experiment}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_config)
    
    print(f"配置文件已生成：{output_file}")

def generate_all_configs():
    """生成所有配置文件"""
    base_dir = os.path.dirname(__file__)
    template_path = os.path.join(base_dir, 'base_template.json')
    network_config_path = os.path.join(base_dir, 'network_configs.json')
    experiment_config_path = os.path.join(base_dir, 'experiment_configs.json')
    output_dir = os.path.join(base_dir, '8x_generated')
    
    # 加载配置
    template = load_template(template_path)
    network_configs = load_network_configs(network_config_path)
    experiment_configs = load_experiment_configs(experiment_config_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成所有配置
    for exp_name, exp_config in experiment_configs['experiments'].items():
        final_config = generate_config(template, exp_config, network_configs, network_config_path)
        output_file = os.path.join(output_dir, f"{exp_name}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_config)
        
        print(f"已生成：{exp_name}.json")

if __name__ == '__main__':
    # 如果没有参数，生成所有配置
    import sys
    if len(sys.argv) == 1:
        print("生成所有配置文件...")
        generate_all_configs()
    else:
        main() 