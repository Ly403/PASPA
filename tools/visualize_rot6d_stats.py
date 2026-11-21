#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rot6D 流形距离统计可视化脚本

该脚本读取采样过程中保存的 rot6d_stats.json 文件，
并绘制各种统计指标随采样步数变化的曲线图。

使用方法：
    python tools/visualize_rot6d_stats.py --json_path outputs/xxx/rot6d_stats/validation/xxx_rot6d_stats.json
    
    # 或者指定目录，自动查找最新的 json 文件
    python tools/visualize_rot6d_stats.py --stats_dir outputs/xxx/rot6d_stats/validation/
    
    # 同时绘制多个实验对比
    python tools/visualize_rot6d_stats.py --json_paths exp1.json exp2.json --labels "Exp1" "Exp2"
"""

import json
import argparse
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_rot6d_stats(json_path: str) -> List[Dict[str, Any]]:
    """加载 rot6d_stats JSON 文件"""
    with open(json_path, "r") as f:
        return json.load(f)


def find_latest_json(stats_dir: str) -> Optional[str]:
    """在目录中找到最新的 rot6d_stats.json 文件"""
    pattern = os.path.join(stats_dir, "*rot6d_stats.json")
    files = glob.glob(pattern)
    if not files:
        return None
    # 按修改时间排序
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def extract_metrics(stats_list: List[Dict[str, Any]], metric_keys: List[str]) -> Dict[str, np.ndarray]:
    """从统计列表中提取指定指标"""
    result = {key: [] for key in metric_keys}
    result["steps"] = []
    result["timesteps"] = []
    
    for stats in stats_list:
        result["steps"].append(stats.get("step", -1))
        result["timesteps"].append(stats.get("timestep", 0))
        for key in metric_keys:
            result[key].append(stats.get(key, np.nan))
    
    for key in result:
        result[key] = np.array(result[key])
    
    return result


def plot_single_experiment(
    stats_list: List[Dict[str, Any]], 
    save_path: Optional[str] = None,
    title_prefix: str = "",
    figsize: tuple = (16, 12),
):
    """
    绘制单个实验的 rot6d 统计曲线
    
    Args:
        stats_list: 统计列表
        save_path: 保存路径
        title_prefix: 标题前缀
        figsize: 图像大小
    """
    # 定义要绘制的指标组
    metric_groups = {
        "Projection Distance": {
            "metrics": ["proj_l2_mean", "proj_l2_max", "proj_l2_min"],
            "labels": ["Mean", "Max", "Min"],
            "colors": ["blue", "red", "green"],
            "ylabel": "L2 Distance",
        },
        "Unit Norm Error": {
            "metrics": ["a1_norm_err_mean", "a2_norm_err_mean"],
            "labels": ["a1 (||a1||-1)", "a2 (||a2||-1)"],
            "colors": ["blue", "orange"],
            "ylabel": "Absolute Error",
        },
        "Orthogonality Error": {
            "metrics": ["orth_err_mean", "orth_err_max"],
            "labels": ["Mean |a1·a2|", "Max |a1·a2|"],
            "colors": ["purple", "red"],
            "ylabel": "Inner Product",
        },
        "Gram-Schmidt Correction": {
            "metrics": ["gs_correction_mean", "gs_correction_max"],
            "labels": ["Mean", "Max"],
            "colors": ["brown", "red"],
            "ylabel": "Correction Magnitude",
        },
        "Model Output (pred_motion)": {
            "metrics": ["pred_proj_l2_mean", "pred_a1_norm_err_mean", "pred_orth_err_mean"],
            "labels": ["Proj L2", "a1 Norm Err", "Orth Err"],
            "colors": ["blue", "orange", "green"],
            "ylabel": "Value",
        },
        "Norm Values": {
            "metrics": ["a1_norm_mean", "a2_norm_mean"],
            "labels": ["||a1||", "||a2||"],
            "colors": ["blue", "orange"],
            "ylabel": "Norm Value",
            "hline": 1.0,  # 理想值是 1
        },
    }
    
    # 提取所有需要的指标
    all_metrics = []
    for group in metric_groups.values():
        all_metrics.extend(group["metrics"])
    all_metrics = list(set(all_metrics))
    
    data = extract_metrics(stats_list, all_metrics)
    steps = data["steps"]
    timesteps = data["timesteps"]
    
    # 创建子图
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # 绘制每个指标组
    for idx, (group_name, group_config) in enumerate(metric_groups.items()):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        for metric, label, color in zip(
            group_config["metrics"], 
            group_config["labels"], 
            group_config["colors"]
        ):
            values = data.get(metric)
            if values is not None and not np.all(np.isnan(values)):
                ax.plot(steps, values, label=label, color=color, marker='o', markersize=4)
        
        ax.set_xlabel("Sampling Step")
        ax.set_ylabel(group_config["ylabel"])
        ax.set_title(f"{title_prefix}{group_name}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 添加水平参考线
        if "hline" in group_config:
            ax.axhline(y=group_config["hline"], color='gray', linestyle='--', alpha=0.5, label=f"Ideal: {group_config['hline']}")
        
        # 设置 x 轴范围
        ax.set_xlim(steps.min() - 0.5, steps.max() + 0.5)
    
    # 添加第二个 x 轴显示 timestep
    for ax in fig.axes:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        # 只显示几个关键点的 timestep
        tick_indices = np.linspace(0, len(steps)-1, min(6, len(steps)), dtype=int)
        ax2.set_xticks(steps[tick_indices])
        ax2.set_xticklabels([f"{timesteps[i]:.0f}" for i in tick_indices], fontsize=8)
        ax2.set_xlabel("Timestep (t)", fontsize=8)
    
    # 整体标题
    fig.suptitle(
        f"Rot6D Manifold Distance Analysis\n(Total {len(stats_list)} sampling steps)",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig


def plot_comparison(
    stats_lists: List[List[Dict[str, Any]]],
    labels: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 10),
):
    """
    绘制多个实验的对比曲线
    
    Args:
        stats_lists: 多个实验的统计列表
        labels: 每个实验的标签
        save_path: 保存路径
        figsize: 图像大小
    """
    # 关键指标对比
    key_metrics = [
        ("proj_l2_mean", "Projection L2 (Mean)"),
        ("a1_norm_err_mean", "a1 Norm Error (Mean)"),
        ("orth_err_mean", "Orthogonality Error (Mean)"),
        ("gs_correction_mean", "GS Correction (Mean)"),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats_lists)))
    
    for idx, (metric_key, metric_name) in enumerate(key_metrics):
        ax = axes[idx // 2, idx % 2]
        
        for stats_list, label, color in zip(stats_lists, labels, colors):
            data = extract_metrics(stats_list, [metric_key])
            steps = data["steps"]
            values = data[metric_key]
            
            if not np.all(np.isnan(values)):
                ax.plot(steps, values, label=label, color=color, marker='o', markersize=3)
        
        ax.set_xlabel("Sampling Step")
        ax.set_ylabel("Value")
        ax.set_title(metric_name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Rot6D Manifold Distance Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    plt.show()
    return fig


def print_summary_table(stats_list: List[Dict[str, Any]], name: str = "Experiment"):
    """打印统计摘要表格"""
    if len(stats_list) < 2:
        print("Not enough data points to generate summary.")
        return
    
    first = stats_list[0]
    last = stats_list[-1]
    
    metrics = [
        ("proj_l2_mean", "Projection L2 (Mean)"),
        ("proj_l2_max", "Projection L2 (Max)"),
        ("a1_norm_err_mean", "a1 Norm Error (Mean)"),
        ("a2_norm_err_mean", "a2 Norm Error (Mean)"),
        ("orth_err_mean", "Orthogonality Error (Mean)"),
        ("gs_correction_mean", "GS Correction (Mean)"),
    ]
    
    print(f"\n{'='*70}")
    print(f" Summary: {name}")
    print(f" Total Steps: {len(stats_list) - 1}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Initial (t=1)':<15} {'Final (t≈0)':<15} {'Change %':<12}")
    print(f"{'-'*70}")
    
    for key, name in metrics:
        if key in first and key in last:
            init_val = first[key]
            final_val = last[key]
            change_pct = ((final_val - init_val) / (init_val + 1e-8)) * 100
            print(f"{name:<30} {init_val:<15.6f} {final_val:<15.6f} {change_pct:+10.1f}%")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Rot6D manifold distance statistics from sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--json_path", 
        type=str, 
        default=None,
        help="Path to a single rot6d_stats.json file"
    )
    parser.add_argument(
        "--stats_dir", 
        type=str, 
        default=None,
        help="Directory containing rot6d_stats.json files (will use the latest)"
    )
    parser.add_argument(
        "--json_paths", 
        type=str, 
        nargs="+",
        default=None,
        help="Multiple JSON paths for comparison"
    )
    parser.add_argument(
        "--labels", 
        type=str, 
        nargs="+",
        default=None,
        help="Labels for each experiment (used with --json_paths)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output path for the figure (default: same directory as JSON with .png extension)"
    )
    parser.add_argument(
        "--no_show", 
        action="store_true",
        help="Don't display the plot (only save)"
    )
    parser.add_argument(
        "--figsize", 
        type=float, 
        nargs=2, 
        default=[16, 12],
        help="Figure size (width height)"
    )
    
    args = parser.parse_args()
    
    # 如果不显示图，使用 Agg 后端
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    
    # 确定要处理的 JSON 文件
    json_paths = []
    
    if args.json_paths:
        json_paths = args.json_paths
    elif args.json_path:
        json_paths = [args.json_path]
    elif args.stats_dir:
        latest = find_latest_json(args.stats_dir)
        if latest:
            json_paths = [latest]
            print(f"Found latest JSON: {latest}")
        else:
            print(f"No rot6d_stats.json files found in {args.stats_dir}")
            return
    else:
        print("Please specify --json_path, --stats_dir, or --json_paths")
        parser.print_help()
        return
    
    # 加载数据
    stats_lists = []
    for path in json_paths:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue
        stats_lists.append(load_rot6d_stats(path))
        print(f"Loaded: {path} ({len(stats_lists[-1])} data points)")
    
    if not stats_lists:
        print("No data loaded.")
        return
    
    # 确定输出路径
    output_path = args.output
    if output_path is None and len(json_paths) == 1:
        output_path = json_paths[0].replace(".json", "_plot.png")
    elif output_path is None:
        output_path = "rot6d_comparison.png"
    
    # 绘图
    if len(stats_lists) == 1:
        # 单个实验
        print_summary_table(stats_lists[0], name=Path(json_paths[0]).stem)
        plot_single_experiment(
            stats_lists[0], 
            save_path=output_path,
            figsize=tuple(args.figsize)
        )
    else:
        # 多个实验对比
        labels = args.labels if args.labels else [Path(p).stem for p in json_paths]
        for stats_list, label in zip(stats_lists, labels):
            print_summary_table(stats_list, name=label)
        plot_comparison(
            stats_lists, 
            labels,
            save_path=output_path,
            figsize=tuple(args.figsize)
        )


if __name__ == "__main__":
    main()

