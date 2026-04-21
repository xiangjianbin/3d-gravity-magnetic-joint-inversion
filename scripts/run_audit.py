"""
实验完整性审计脚本
==================

检查项:
  A. 真实标签来源 -- 数据集 GT 是合成的还是从模型输出派生的？
  B. 分数归一化   -- 指标是否除以自身最大值？（人为抬高分数的常见手段）
  C. 结果文件存在性 -- 声称的文件是否都存在？
  D. 死程代码     -- 定义的指标函数是否实际被调用？
  E. 范围评估     -- 测试了多少样本？能否支撑论文声明？

输出:
  docs/EXPERIMENT_AUDIT.md   (Markdown 报告)
  docs/EXPERIMENT_AUDIT.json (结构化结果)

用法:
    python scripts/run_audit.py [--project-root /path/to/project]

作者: 审计 Agent
日期: 2026-04-21
"""

import os
import sys
import json
import re
import ast
import time
from pathlib import Path
from collections import OrderedDict

# ============================================================
# 项目路径配置
# ============================================================

DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 论文声称的关键数值（用于对标）
PAPER_CLAIMS = {
    "total_samples": 45000,
    "train_samples": 31500,
    "val_samples": 9000,
    "test_samples": 4500,
    "dataset_types": 6,
    "n_epochs_converge": 60,       # 论文 Fig.5 推断收敛于 60-80 epoch
    "model_params_approx": 100e6,  # 约 100M 参数
}

# 审计时期望存在的关键文件清单
EXPECTED_FILES = {
    "docs/PAPER_ANALYSIS.md":              "论文解析报告",
    "docs/DATASET_REPORT.md":               "数据集生成报告",
    "docs/MODEL_REPORT.md":                 "模型架构报告",
    "docs/SMOKE_TEST_REPORT.md":            "冒烟测试报告",
    "docs/TRAINING_LOG.md":                 "训练日志",
    "data/train_dataset.npz":               "训练数据集",
    "data/val_dataset.npz":                 "验证数据集",
    "data/test_dataset.npz":                "测试数据集",
    "checkpoints/best_model.pt":            "最佳模型权重",
    "results/metrics.json":                 "评估指标 JSON",
    "src/model/joint_inversion_net.py":     "网络组装代码",
    "src/model/task_heads.py":              "任务头定义",
    "src/model/loss_functions.py":          "损失函数",
    "src/data/generate_synthetic.py":       "合成数据生成器",
    "src/data/dataset.py":                  "Dataset 类",
    "src/train.py":                         "训练入口",
    "src/evaluate.py":                      "评估入口",
    "src/utils.py":                         "工具函数",
}


def resolve_path(project_root: str, relative: str) -> str:
    """拼接绝对路径。"""
    return os.path.join(project_root, relative)


def file_exists(path: str) -> bool:
    """检查文件是否存在。"""
    return os.path.isfile(path)


def read_file_safe(path: str, default: str = "") -> str:
    """安全读取文件，不存在则返回默认值。"""
    if not file_exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def read_json_safe(path: str) -> dict:
    """安全读取 JSON 文件。"""
    if not file_exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ============================================================
# 审计检查 A: 真实标签来源
# ============================================================

def check_gt_source(project_root: str) -> dict:
    """
    检查 A: GT 标签是否为独立合成数据（而非模型输出派生）。

    关键信号:
      - generate_synthetic.py 中 rho/kappa 是否独立于任何模型输出生成
      - 数据保存流程中是否有 model inference 步骤
      - structural_sim 标签是否由 rho!=0 & kappa!=0 规则直接计算
    """
    result = {
        "check_id": "A",
        "title": "真实标签来源审计",
        "verdict": "PASS",
        "details": [],
        "risk_level": "LOW",
    }

    gen_path = resolve_path(project_root, "src/data/generate_synthetic.py")
    ds_path = resolve_path(project_root, "src/data/dataset.py")
    gen_code = read_file_safe(gen_path)
    ds_code = read_file_safe(ds_path)

    if not gen_code:
        result["verdict"] = "WARN"
        result["details"].append("generate_synthetic.py 不存在，无法验证 GT 来源")
        result["risk_level"] = "HIGH"
        return result

    # 检查 1: GT 是否由地质体模型函数独立生成
    gt_gen_patterns = [
        ("generate_cuboid", "长方体地质体生成"),
        ("generate_tilting_body", "倾斜体地质体生成"),
        ("generate_random_walk_body", "随机游走地质体生成"),
        ("compute_structural_similarity", "结构相似性标签规则计算"),
    ]
    found_patterns = []
    for pattern, desc in gt_gen_patterns:
        if pattern in gen_code:
            found_patterns.append(desc)
    if found_patterns:
        result["details"].append(
            f"GT 由以下独立地质体模型函数生成: {', '.join(found_patterns)}"
        )

    # 检查 2: 是否有任何 model / inference / predict 调用出现在数据生成流程中
    suspicious_keywords = [
        r"model\s*\.\s*(forward|predict|infer)",
        r"\.load_state_dict",
        r"torch\.load\(.*checkpoint",
        r"model\(.*input\)",
        r"with torch\.no_grad\(\).*model",
    ]
    model_refs_in_gen = []
    for kw in suspicious_keywords:
        matches = re.findall(kw, gen_code)
        if matches:
            model_refs_in_gen.append(kw)
    if model_refs_in_gen:
        result["verdict"] = "FAIL"
        result["risk_level"] = "CRITICAL"
        result["details"].append(
            f"[严重] 数据生成代码中发现模型推理引用: {model_refs_in_gen}。"
            "GT 可能是从模型输出派生的，而非独立合成。"
        )
    else:
        result["details"].append("数据生成代码中未发现模型推理调用，GT 为独立合成")

    # 检查 3: structural_sim 标签规则
    sim_rule_pattern = r"rho.*!=\s*0.*&.*kappa.*!=\s*0"
    if re.search(sim_rule_pattern, gen_code):
        result["details"].append(
            "结构相似性标签 S 由规则 `rho!=0 AND kappa!=0` 直接计算，非模型输出"
        )
    else:
        result["verdict"] = "WARN"
        result["details"].append(
            "[警告] 未找到标准的结构相似性标签生成规则，需人工确认"
        )
        result["risk_level"] = "MEDIUM"

    # 检查 4: 正演引擎类型
    if "simple_forward_gravity" in gen_code and "HAS_EXTERNAL_FORWARD" in gen_code:
        if 'HAS_EXTERNAL_FORWARD = True' in gen_code or \
           "from src.data.forward_gravity import" in gen_code:
            result["details"].append("正演引擎: 外部物理模块 (Blakely 公式)")
        else:
            result["details"].append(
                "正演引擎: 内置简化版 (深度加权投影)，非严格 Blakely 公式。"
                "这可能影响数据真实性但不会造成 GT 泄露。"
            )
            if result["risk_level"] == "LOW":
                result["risk_level"] = "MEDIUM"

    return result


# ============================================================
# 审计检查 B: 分数归一化
# ============================================================

def check_score_normalization(project_root: str) -> dict:
    """
    检查 B: 评估指标是否存在人为归一化（除以自身最大值等）。

    常见作弊模式:
      - metric /= metric.max()  或类似操作
      - 将 MSE 等指标映射到 [0,1] 后再报告
      - 只报告归一化后的相关系数而不报告绝对误差
    """
    result = {
        "check_id": "B",
        "title": "分数归一化审计",
        "verdict": "PASS",
        "details": [],
        "risk_level": "LOW",
    }

    utils_path = resolve_path(project_root, "src/utils.py")
    eval_path = resolve_path(project_root, "src/evaluate.py")
    utils_code = read_file_safe(utils_path)
    eval_code = read_file_safe(eval_path)

    # 可疑归一化模式
    suspicious_norm_patterns = [
        (r"\s*\/=\s*\w+\.max\(\)", "除以自身最大值"),
        (r"\s*\/=\s*\w+\.abs\(\)\.max\(\)", "除以绝对值最大值"),
        (r"normalized\s*=\s*.*metric", "变量名含 normalized 的指标处理"),
        (r"scale\s*=\s*.*100\s*/", "缩放到百分比的指标处理"),
        (r"metric\s*\*\s*=\s*100", "乘以 100 的指标处理"),
        (r"\.clip\(\s*0\s*,\s*1\s*\)", "裁剪到 [0,1] 的指标处理"),
    ]

    files_to_check = [
        ("src/utils.py", utils_code),
        ("src/evaluate.py", eval_code),
    ]

    all_suspicious = []
    for fname, code in files_to_check:
        if not code:
            continue
        for pattern, desc in suspicious_norm_patterns:
            matches = re.findall(pattern, code)
            if matches:
                line_nums = []
                for i, line in enumerate(code.split("\n"), 1):
                    if re.search(pattern, line):
                        line_nums.append(i)
                all_suspicious.append(
                    f"{fname}:{line_nums}: 发现 `{desc}` 模式"
                )

    if all_suspicious:
        result["verdict"] = "WARN"
        result["risk_level"] = "MEDIUM"
        result["details"].append(
            f"发现可能的归一化操作:\n  - " + "\n  - ".join(all_suspicious)
        )
    else:
        result["details"].append(
            "utils.py 和 evaluate.py 中未发现可疑的指标归一化操作"
        )

    # 检查 compute_metrics 函数的具体实现
    if "compute_metrics" in utils_code:
        # 提取 compute_metrics 函数体
        func_match = re.search(
            r"def compute_metrics\(.*?\):(.*?)(?=\ndef |\nclass |\Z)",
            utils_code, re.DOTALL
        )
        if func_match:
            func_body = func_match.group(1)
            # 检查标准指标实现
            standard_indicators = ["mse =", "rmse =", "mae =", "correlation"]
            found_standard = [ind for ind in standard_indicators if ind in func_body]
            result["details"].append(
                f"compute_metrics 包含标准指标实现: {', '.join(found_standard)}"
            )
            # 检查是否有后处理
            if "/=" in func_body or "*=" in func_body:
                post_ops = [line.strip() for line in func_body.split("\n")
                            if ("/=" in line or "*=" in line)]
                result["details"].append(
                    f"[注意] compute_metrics 中有赋值运算: {post_ops}"
                )

    return result


# ============================================================
# 审计检查 C: 结果文件存在性
# ============================================================

def check_file_existence(project_root: str) -> dict:
    """
    检查 C: 关键产出文件是否都存在。
    """
    result = {
        "check_id": "C",
        "title": "结果文件存在性审计",
        "verdict": "PASS",
        "details": [],
        "file_table": {},
        "risk_level": "LOW",
    }

    present = 0
    missing = 0
    missing_critical = []

    for rel_path, description in EXPECTED_FILES.items():
        abs_path = resolve_path(project_root, rel_path)
        exists = file_exists(abs_path)
        result["file_table"][rel_path] = {
            "description": description,
            "exists": exists,
            "path": abs_path,
        }
        if exists:
            present += 1
            size_kb = os.path.getsize(abs_path) / 1024
            result["details"].append(f"  [OK]   {rel_path} ({size_kb:.1f} KB)")
        else:
            missing += 1
            result["details"].append(f"  [MISS] {rel_path}")
            # 判断缺失严重程度
            critical_files = [
                "checkpoints/best_model.pt",
                "results/metrics.json",
                "src/model/joint_inversion_net.py",
                "src/train.py",
            ]
            if rel_path in critical_files:
                missing_critical.append(rel_path)

    total = len(EXPECTED_FILES)
    result["summary"] = f"{present}/{total} 文件存在"

    if missing_critical:
        result["verdict"] = "FAIL"
        result["risk_level"] = "CRITICAL"
        result["details"].append(
            f"\n[严重] 缺少关键文件: {', '.join(missing_critical)}"
        )
    elif missing > 3:
        result["verdict"] = "WARN"
        result["risk_level"] = "MEDIUM"
        result["details"].append(f"\n[警告] {missing}/{total} 文件缺失")
    elif missing > 0:
        result["verdict"] = "WARN"
        result["risk_level"] = "LOW"
        result["details"].append(f"\n[信息] {missing}/{total} 非关键文件缺失")

    return result


# ============================================================
# 审计检查 D: 死程代码
# ============================================================

def check_dead_code(project_root: str) -> dict:
    """
    检查 D: 定义了但未被调用的指标/函数。

    方法:
      1. 扫描所有 .py 文件中的 def 开头的函数定义
      2. 在项目范围内搜索每个函数名的调用
      3. 标记只定义不调用的函数
    """
    result = {
        "check_id": "D",
        "title": "死程代码审计",
        "verdict": "PASS",
        "details": [],
        "dead_functions": [],
        "risk_level": "LOW",
    }

    src_dir = resolve_path(project_root, "src")
    scripts_dir = resolve_path(project_root, "scripts")

    # 收集所有 Python 源码文本
    all_code_text = ""
    all_py_files = []

    for search_dir in [src_dir, scripts_dir]:
        if not os.path.isdir(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            # 跳过 __pycache__
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            code = f.read()
                        all_code_text += code + "\n"
                        all_py_files.append(fpath)
                    except Exception:
                        pass

    if not all_code_text:
        result["verdict"] = "WARN"
        result["details"].append("未找到任何 Python 源码文件")
        return result

    # 提取所有顶层函数定义
    func_defs = {}  # name -> defined_in_file
    for fpath in all_py_files:
        code = read_file_safe(fpath)
        for match in re.finditer(r"^def (\w+)\s*\(", code, re.MULTILINE):
            fname = match.group(1)
            # 跳过特殊方法、私有方法和测试方法
            if fname.startswith("_") and not fname.startswith("__"):
                continue
            if fname in ("main",):  # 入口函数不算死代码
                continue
            if fname not in func_defs:
                func_defs[fname] = os.path.relpath(fpath, project_root)

    # 对每个函数检查是否被调用
    dead_funcs = []
    for func_name, defined_in in sorted(func_defs.items()):
        # 搜索函数名作为标识符出现的情况（排除定义行本身）
        # 使用词边界匹配避免子串误判
        pattern = r'\b' + re.escape(func_name) + r'\b'
        all_matches = re.findall(pattern, all_code_text)

        # 如果出现次数 <= 1（只有定义），则可能是死代码
        # 但要排除一些特殊情况: 装饰器、字符串引用、类方法
        if len(all_matches) <= 1:
            # 进一步确认：搜索赋值/调用上下文
            call_pattern = rf'{re.escape(func_name)}\s*\('
            call_matches = re.findall(call_pattern, all_code_text)
            if len(call_matches) == 0:
                dead_funcs.append({
                    "name": func_name,
                    "defined_in": defined_in,
                })

    if dead_funcs:
        result["dead_functions"] = dead_funcs
        # 过滤掉明显不是问题的（如 __main__ 保护下的辅助函数）
        truly_dead = []
        possibly_ok = []
        for df in dead_funcs:
            # 某些函数可能是通过其他方式间接调用的（如注册到某个列表中）
            # 这里做简单启发式判断
            name = df["name"]
            if name in ("set_seed", "count_parameters", "save_checkpoint",
                        "load_checkpoint", "AverageMeter", "compute_metrics"):
                # 这些在 utils.py 定义但在 train.py/evaluate.py 导入使用
                # import 语句可能不包含括号调用形式
                import_pattern = rf'(?:from|import)\s+.*\b{re.escape(name)}\b'
                if re.search(import_pattern, all_code_text):
                    possibly_ok.append(df)
                    continue
            truly_dead.append(df)

        if truly_dead:
            result["verdict"] = "WARN"
            result["risk_level"] = "LOW"
            result["details"].append(
                f"发现 {len(truly_dead)} 个可能未被调用的函数:\n" +
                "\n".join(f"  - {df['name']} (定义于 {df['defined_in']})"
                          for df in truly_dead)
            )
        elif possibly_ok:
            result["details"].append(
                f"所有定义的函数均有导入或调用记录 "
                f"({len(possibly_ok)} 个仅通过 import 引用)"
            )
    else:
        result["details"].append("所有定义的函数均被发现有调用或导入记录")

    return result


# ============================================================
# 审计检查 E: 范围评估
# ============================================================

def check_evaluation_scope(project_root: str) -> dict:
    """
    检查 E: 测试样本量是否足以支撑论文声明。

    论文声称:
      - 总样本 45,000
      - 测试集 4,500 (10%)
      - 6 类数据集
      - 8 个测试源体 (Table III)
    """
    result = {
        "check_id": "E",
        "title": "范围评估审计",
        "verdict": "PASS",
        "details": [],
        "stats": {},
        "risk_level": "LOW",
    }

    # 1. 从 DATASET_REPORT.md 提取实际样本数
    dataset_report = read_file_safe(resolve_path(project_root, "docs/DATASET_REPORT.md"))

    actual_total = None
    actual_train = None
    actual_val = None
    actual_test = None

    if dataset_report:
        # 尝试提取合计数
        total_match = re.search(r'\*\*合计\*\*.*?\|\s*(\d+)', dataset_report)
        if total_match:
            actual_total = int(total_match.group(1))

        # 尝试提取划分数量
        train_match = re.search(r'train_dataset\.npz.*?(\d+)\s*samples?', dataset_report)
        val_match = re.search(r'val_dataset\.npz.*?(\d+)\s*samples?', dataset_report)
        test_match = re.search(r'test_dataset\.npz.*?(\d+)\s*samples?', dataset_report)
        if train_match:
            actual_train = int(train_match.group(1))
        if val_match:
            actual_val = int(val_match.group(1))
        if test_match:
            actual_test = int(test_match.group(1))

    # 2. 直接从 npz 文件获取更精确的数据
    for split in ["train", "val", "test"]:
        npz_path = resolve_path(project_root, f"data/{split}_dataset.npz")
        if file_exists(npz_path):
            try:
                data = np.load(npz_path, allow_pickle=True)
                meta_str = str(data['__meta__'])
                meta = json.loads(meta_str)
                n = meta.get('n_samples', 0)
                result["stats"][f"{split}_npz_samples"] = n
                data.close()
            except Exception as e:
                result["details"].append(f"  读取 {npz_path} 失败: {e}")

    # 3. 从 metrics.json 获取测试时使用的样本数
    metrics = read_json_safe(resolve_path(project_root, "results/metrics.json"))
    if metrics:
        config = metrics.get("config", {})
        n_eval = config.get("n_samples")
        if n_eval is not None:
            result["stats"]["eval_n_samples"] = n_eval
        eval_split = config.get("split", "N/A")
        result["stats"]["eval_split"] = eval_split

    # 4. 从 training_history.json 获取训练轮次
    history_path = resolve_path(project_root, "results/training_history.json")
    if file_exists(history_path):
        history = read_json_safe(history_path)
        if history:
            result["stats"]["training_epochs_completed"] = len(history)
            last_epoch = history[-1].get("epoch", "N/A") if history else "N/A"
            result["stats"]["last_epoch"] = last_epoch

    # 5. 汇总对比
    paper_total = PAPER_CLAIMS["total_samples"]
    paper_test = PAPER_CLAIMS["test_samples"]
    paper_epochs = PAPER_CLAIMS["n_epochs_converge"]

    actual_total_final = actual_total or result["stats"].get("train_npz_samples", 0) \
                         + result["stats"].get("val_npz_samples", 0) \
                         + result["stats"].get("test_npz_samples", 0)
    actual_test_final = actual_test or result["stats"].get("test_npz_samples", 0)
    actual_epochs = result["stats"].get("training_epochs_completed", 0)

    result["stats"]["paper_claimed_total"] = paper_total
    result["stats"]["actual_total"] = actual_total_final
    result["stats"]["paper_claimed_test"] = paper_test
    result["stats"]["actual_test"] = actual_test_final
    result["stats"]["paper_claimed_min_epochs"] = paper_epochs
    result["stats"]["actual_epochs"] = actual_epochs

    # 判定
    details_lines = []

    # 样本量对比
    if actual_total_final > 0:
        ratio = actual_total_final / paper_total * 100
        details_lines.append(
            f"总样本量: 实际={actual_total_final}, 论文声称={paper_total} "
            f"(实际/论文={ratio:.1f}%)"
        )
        if ratio < 1.0:
            details_lines.append(
                f"  -> 样本量为论文的 {ratio:.1f}%，属于小规模验证实验"
            )
            result["risk_level"] = "MEDIUM"
            result["verdict"] = "WARN"
        else:
            details_lines.append("  -> 样本量达到论文规模")
    else:
        details_lines.append("总样本量: N/A (无法确定)")
        result["verdict"] = "WARN"
        result["risk_level"] = "MEDIUM"

    # 测试集对比
    if actual_test_final > 0:
        test_ratio = actual_test_final / paper_test * 100 if paper_test > 0 else 0
        details_lines.append(
            f"测试集: 实际={actual_test_final}, 论文声称={paper_test} "
            f"(实际/论文={test_ratio:.1f}%)"
        )
        if actual_test_final < 30:
            details_lines.append(
                "  -> [警告] 测试样本少于 30，统计显著性不足"
            )
            result["risk_level"] = max(result["risk_level"], "MEDIUM", key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x))
            result["verdict"] = "WARN"

    # 训练轮次对比
    if actual_epochs > 0:
        details_lines.append(
            f"训练轮次: 实际={actual_epochs}, 论文推断收敛约 {paper_epochs} epoch"
        )
        if actual_epochs < 10:
            details_lines.append(
                "  -> [警告] 训练轮次过少 (<10)，模型可能未充分收敛"
            )
            result["verdict"] = "WARN"
            result["risk_level"] = max(result["risk_level"], "MEDIUM", key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x))
        elif actual_epochs < paper_epochs * 0.5:
            details_lines.append(
                f"  -> 训练轮次少于论文收敛轮次的 50%，结果可能不可比"
            )
            result["verdict"] = "WARN"
            result["risk_level"] = max(result["risk_level"], "MEDIUM", key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x))
    else:
        details_lines.append("训练轮次: N/A (无训练历史记录)")
        result["verdict"] = "WARN"

    result["details"] = details_lines

    return result


# ============================================================
# 综合判定
# ============================================================

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


def overall_verdict(checks: list) -> tuple:
    """
    根据各检查项给出综合判定。

    Returns:
        (verdict, summary)
    """
    fail_count = sum(1 for c in checks if c["verdict"] == "FAIL")
    warn_count = sum(1 for c in checks if c["verdict"] == "WARN")
    pass_count = sum(1 for c in checks if c["verdict"] == "PASS")

    max_risk = "LOW"
    for c in checks:
        if RISK_ORDER.get(c["risk_level"], 0) > RISK_ORDER.get(max_risk, 0):
            max_risk = c["risk_level"]

    if fail_count > 0:
        verdict = "FAIL"
        summary = f"{fail_count} 项 FAIL, {warn_count} 项 WARN, {pass_count} 项 PASS"
    elif warn_count >= 3:
        verdict = "WARN"
        summary = f"全部 PASS/WARN ({warn_count} WARN)。建议修复后再下结论"
    elif warn_count > 0:
        verdict = "PASS_WITH_NOTES"
        summary = f"基本通过 ({warn_count} 项需关注)"
    else:
        verdict = "PASS"
        summary = "全部检查项通过"

    return verdict, summary, max_risk


# ============================================================
# Markdown 报告生成
# ============================================================

def generate_markdown_report(checks: list, overall: tuple, project_root: str) -> str:
    """生成 EXPERIMENT_AUDIT.md 内容。"""
    verdict, summary, max_risk = overall
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# 实验完整性审计报告\n")
    lines.append(f"> **审计时间**: {timestamp}")
    lines.append(f"> **项目路径**: {project_root}")
    lines.append(f"> **综合判定**: **{verdict}** | 风险等级: **{max_risk}**")
    lines.append(f"> **摘要**: {summary}\n")

    lines.append("---\n")

    # 各检查项详情
    for check in checks:
        icon = {"PASS": "OK", "WARN": "!!", "FAIL": "XX"}.get(check["verdict"], "??")
        lines.append(f"## 检查 {check['check_id']}: {check['title']} [{icon}]\n")
        lines.append(f"- **判定**: {check['verdict']}")
        lines.append(f"- **风险等级**: {check['risk_level']}")
        lines.append("- **详情**:")
        for detail in check.get("details", []):
            lines.append(f"  - {detail}")

        # 检查 C 的文件表
        if check["check_id"] == "C" and "file_table" in check:
            lines.append("\n### 文件清单\n")
            lines.append("| 文件 | 说明 | 状态 |")
            lines.append("|------|------|------|")
            for fp, info in check["file_table"].items():
                status = "OK" if info["exists"] else "MISSING"
                lines.append(f"| `{fp}` | {info['description']} | {status} |")

        # 检查 E 的统计表
        if check["check_id"] == "E" and "stats" in check:
            lines.append("\n### 范围统计\n")
            lines.append("| 指标 | 论文声称 | 实际值 |")
            lines.append("|------|---------|--------|")
            stat_display = {
                "paper_claimed_total": "总样本量",
                "actual_total": "实际总样本",
                "paper_claimed_test": "测试集大小",
                "actual_test": "实际测试集",
                "paper_claimed_min_epochs": "最小收敛轮次",
                "actual_epochs": "实际训练轮次",
                "eval_n_samples": "评估样本数",
                "eval_split": "评估数据划分",
            }
            for key, label in stat_display.items():
                val = check["stats"].get(key, "N/A")
                lines.append(f"| {label} | {PAPER_CLAIMS.get({'paper_claimed_total': 'total_samples','paper_claimed_test': 'test_samples','paper_claimed_min_epochs': 'n_epochs_converge'}.get(key, ''), 'N/A')} | {val} |")

        lines.append("")

    # 结论
    lines.append("---\n")
    lines.append("## 审计结论\n")

    risk_verdict_map = {
        ("PASS", "LOW"): (
            "实验流程完整，未发现数据泄露或人为干预痕迹。",
            "可基于当前结果进行结论判定。"
        ),
        ("PASS_WITH_NOTES", "LOW"): (
            "实验流程基本完整，有少量注意事项。",
            "建议关注 WARN 项后进行结论判定。"
        ),
        ("PASS_WITH_NOTES", "MEDIUM"): (
            "实验流程基本完整，但存在需要关注的偏差。",
            "主要问题: 样本量/训练规模与论文不一致，定量对标需谨慎。"
        ),
        ("WARN", "MEDIUM"): (
            "实验存在一定局限性，部分检查项未完全通过。",
            "建议: 补充缺失文件、扩大测试集后再重新审计。"
        ),
        ("WARN", "HIGH"): (
            "实验存在显著问题，结果可信度存疑。",
            "必须修复 FAIL/WARN 项后方可进行结论判定。"
        ),
        ("FAIL", "CRITICAL"): (
            "实验存在严重问题，结果不可信。",
            "必须修复所有 CRITICAL 问题并重新运行完整 pipeline。"
        ),
    }

    conclusion, recommendation = risk_verdict_map.get(
        (verdict, max_risk),
        ("请人工审查各检查项详情后给出结论。",
         "根据具体 FAIL/WARN 内容决定后续行动。")
    )

    lines.append(f"**结论**: {conclusion}\n")
    lines.append(f"**建议**: {recommendation}\n")

    return "\n".join(lines)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="3D 重磁联合反演实验完整性审计")
    parser.add_argument("--project-root", type=str, default=DEFAULT_PROJECT_ROOT,
                        help="项目根目录路径")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认: <project-root>/docs)")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    output_dir = args.output_dir or os.path.join(project_root, "docs")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("3D 重磁联合反演 -- 实验完整性审计")
    print(f"项目路径: {project_root}")
    print("=" * 60)

    # 执行 5 项检查
    checks = []

    print("\n[A] 真实标签来源审计...")
    checks.append(check_gt_source(project_root))
    print(f"  => {checks[-1]['verdict']} ({checks[-1]['risk_level']})")

    print("\n[B] 分数归一化审计...")
    checks.append(check_score_normalization(project_root))
    print(f"  => {checks[-1]['verdict']} ({checks[-1]['risk_level']})")

    print("\n[C] 结果文件存在性审计...")
    checks.append(check_file_existence(project_root))
    print(f"  => {checks[-1]['verdict']} ({checks[-1]['risk_level']})")
    print(f"  => {checks[-1].get('summary', '')}")

    print("\n[D] 死程代码审计...")
    checks.append(check_dead_code(project_root))
    print(f"  => {checks[-1]['verdict']} ({checks[-1]['risk_level']})")

    print("\n[E] 范围评估审计...")
    checks.append(check_evaluation_scope(project_root))
    print(f"  => {checks[-1]['verdict']} ({checks[-1]['risk_level']})")

    # 综合判定
    verdict, summary, max_risk = overall_verdict(checks)

    print("\n" + "=" * 60)
    print(f"综合判定: {verdict}")
    print(f"风险等级: {max_risk}")
    print(f"摘要:     {summary}")
    print("=" * 60)

    # 生成 Markdown 报告
    md_content = generate_markdown_report(checks, (verdict, summary, max_risk), project_root)
    md_path = os.path.join(output_dir, "EXPERIMENT_AUDIT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\nMarkdown 报告: {md_path}")

    # 生成 JSON 报告
    json_output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": project_root,
        "overall": {
            "verdict": verdict,
            "summary": summary,
            "max_risk_level": max_risk,
        },
        "checks": checks,
    }
    json_path = os.path.join(output_dir, "EXPERIMENT_AUDIT.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"JSON 报告:     {json_path}")

    # 返回退出码
    if verdict == "FAIL":
        sys.exit(1)
    elif verdict in ("WARN", "PASS_WITH_NOTES"):
        sys.exit(0)
    else:
        sys.exit(0)


if __name__ == "__main__":
    import argparse
    main()
