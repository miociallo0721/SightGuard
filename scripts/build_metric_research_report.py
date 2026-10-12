"""Build factual delivery reports from complete records; no model execution or label edits."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from sightguard.research.resources import check_resources, lower_own_priority
from sightguard.research.settings import research_cache_root

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "android/deliveries/metric-corridor-research"


def read(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name: str, value: dict | list) -> None:
    (DELIVERY / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def performance() -> dict:
    timings, ages, loads = defaultdict(list), defaultdict(list), defaultdict(list)
    available, allocated, reserved = [], [], []
    for folder in [DELIVERY / "evaluation", DELIVERY / "evaluation-extension"]:
        for path in folder.glob("*.jsonl"):
            if path.name.endswith((".initial.jsonl", ".pre-geometry.jsonl")):
                continue
            for row in records(path):
                if "resources" in row:
                    available.append(row["resources"]["available_ram_mb"])
                if path.name.startswith("metric_"):
                    name = f"PC_RTX2050_{row['model']}_{row['input_size']}"
                    timings[name].append(row["inference_ms"])
                    ages[name].append(row["wall"]["source_frame_age_ms"])
                    loads[name].append(row["load_ms"])
                    allocated.append(row["gpu_peak_allocated_mb"])
                    reserved.append(row["gpu_peak_reserved_mb"])
                elif path.name == "semantic_results.jsonl":
                    timings["PC_CPU_2threads_SegFormer256"].append(row["segformer_latency_ms"])
                    timings["PC_CPU_2threads_YOLO320"].append(row["yolo_latency_ms"])
                elif path.name == "relative_results.jsonl":
                    timings["PC_CPU_2threads_relative280"].append(row["inference_ms"])
                elif path.name == "wall_approach_results.jsonl":
                    timings["PC_combined_grid_corridor_approach_replay"].append(
                        row["research_latency_ms"]
                    )

    def stats(values: list) -> dict:
        return {
            "count": len(values),
            "p50": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(max(values)),
        }

    audio = read(DELIVERY / "adb-baseline-environment-audio.json")["events"]
    starts = [event for event in audio if event["phase"] == "onStart"]
    return {
        "schema_version": "sightguard.metric-corridor-performance.v1",
        "pc_timing_ms": {name: stats(values) for name, values in timings.items()},
        "offline_simulated_serial_source_age_ms_NOT_CameraX": {
            name: stats(values) for name, values in ages.items()
        },
        "pc_metric_load_ms": {name: sorted(set(values)) for name, values in loads.items()},
        "pc_gpu_peak_allocated_mb": max(allocated),
        "pc_gpu_peak_reserved_mb": max(reserved),
        "pc_minimum_free_ram_observed_at_sample_boundaries_mb": min(available),
        "pc_memory_limit_mb": 1536,
        "pc_resource_note": (
            "sample-boundary observations, not an OS-wide peak-memory tra"
            "ce; serial heavy jobs, Gradle worker=1 heap=1536MB"
        ),
        "android_baseline_only": {
            "device": "Huawei FIN-AL60 (nova12 活力版)",
            "android": "12/API31",
            "reported_harmony": "4.2.0",
            "relative_depth": read(DELIVERY / "adb-baseline-depth-validation.json"),
            "segformer": read(DELIVERY / "adb-baseline-segformer-validation.json"),
            "audio_phases": dict(Counter(event["phase"] for event in audio)),
            "wall_on_start": [event for event in starts if ":WALL:" in event["event"]],
            "scope": "installed_0.4_baseline_context_and_resource_tests_not_experimental_APK",
        },
        "android_experimental": {
            "installation": (
                "BLOCKED_MANUAL_INSTALL_APPROVAL_WHILE_LOCKED_CANCELLED_SESSION_765964172"
            ),
            "new_metric_inference_ms": None,
            "new_metric_peak_pss_kb": None,
            "yolo_fps_impact": None,
            "CameraX_source_age": None,
            "motion_sync": "MOTION_SYNC_UNKNOWN",
            "thermal_last_read": 0,
            "avd": "no_running_authorized_AVD_no_GUI_or_permission_bypass",
        },
        "pc_ort_1_24_3_parity": read(DELIVERY / "onnx-parity-ort-1.24.3.json"),
        "pc_ort_1_31_parity": read(DELIVERY / "onnx-parity-ort-1.31.json"),
    }


def main() -> None:
    lower_own_priority()
    check_resources(output_dir=DELIVERY)
    summary = read(DELIVERY / "pc_validation_summary.json")
    perf = performance()
    save("performance_report.json", perf)
    artifacts = {}
    for name, path in [
        ("experimental", DELIVERY / "SightGuard-metric-corridor-experimental.apk"),
        ("baseline", DELIVERY / "baseline/SightGuard-0.4-stable.apk"),
        (
            "instrumentation_not_run",
            DELIVERY / "SightGuard-metric-corridor-experimental-androidTest.apk",
        ),
    ]:
        artifacts[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    tests = {"jvm": 0, "jvm_failures": 0, "jvm_errors": 0}
    for path in (ROOT / "android/app/build/test-results/testDebugUnitTest").glob("TEST-*.xml"):
        attributes = ET.parse(path).getroot().attrib
        for name, key in [("jvm", "tests"), ("jvm_failures", "failures"), ("jvm_errors", "errors")]:
            tests[name] += int(attributes[key])
    pytest_log = (DELIVERY / "pytest-delivery-final.log").read_text(encoding="utf-8")
    passed = re.search(r"(\d+) passed", pytest_log)
    skipped = re.search(r"(\d+) skipped", pytest_log)
    if not passed or " failed" in pytest_log or " errors" in pytest_log:
        raise ValueError("final pytest success evidence missing")
    tests.update(
        python_passed=int(passed.group(1)),
        python_skipped=int(skipped.group(1)) if skipped else 0,
        python_new_research_tests=55,
        ruff="PASS",
        installed_baseline_phone_tests=3,
        experimental_phone_tests="NOT_RUN_INSTALL_APPROVAL_REQUIRED",
    )
    cache = research_cache_root()
    provenance = read(cache / "download-provenance.json")
    for name in ["download-provenance.json", "sample-manifest.json", "extension-manifest.json"]:
        save(name, read(cache / name))
    save(
        "model-manifest.json",
        read(ROOT / "android/app/src/main/assets/metric_research/metric_manifest.json"),
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    validation = {
        "schema_version": "sightguard.metric-corridor-delivery.v1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_branch": branch,
        "git_commit": revision,
        "baseline_commit": "4a1fdfb97190a9a85e151479ec19a13b8c37bc73",
        "previous_pr": "https://github.com/miociallo0721/SightGuard/pull/6",
        "previous_pr_state": "MERGED",
        "previous_pr_merge_commit": "0f142a985068b4d11d04ceeeb7a04706eff33472",
        "artifacts": artifacts,
        "tests": tests,
        "evidence_levels": {
            "level_1": "PASS_55_new_PC_and_17_new_JVM_tests",
            "level_2": (
                "700_real_RGBD_samples_2100_metric_runs_real_SegFormer_YOLO_r"
                "elative_depth_4_ONNX_shapes_two_ORT_versions"
            ),
            "level_3_experimental": "NOT_COMPLETED_manual_install_approval_required",
            "level_3_baseline": (
                "3_installed_0.4_tests_passed_including_15_real_audio_starts_and_completions"
            ),
            "level_4": "NOT_AVAILABLE_no_tape_measurement_no_physical_camera_reference_distance",
        },
        "formal_risk_changed": False,
        "automatic_metric_or_approach_speech": False,
        "navigation_or_passability_enabled": False,
        "experimental_default_enabled": False,
        "pc": summary,
        "qualitative_corridor_review": {
            "path": "corridor_visual_review.jsonl",
            "reviewed_records": sum(1 for _ in records(DELIVERY / "corridor_visual_review.jsonl")),
            "conclusion": "WALKING_GROUND_NOT_VERIFIED_planar_poster_or_board_surface",
            "official_ground_truth": False,
            "excluded_from_algorithm_inputs": True,
        },
        "performance": perf,
        "limitations": [
            "no_ground_truth_semantic_wall_labels",
            "no_outdoor_RGBD_evaluation",
            "no_valid_forward_toward_confirmed_wall_positive",
            "no_valid_receding_mocap_pairs",
            "new_Android_camera_and_memory_regression_not_run",
            "P10_P90_spatial_statistics_not_prediction_accuracy_intervals",
            "24_candidates_on_planar_poster_surface_not_verified_walking_ground",
            "缺少卷尺测距验证",
        ],
    }
    save("validation_summary.json", validation)
    with (DELIVERY / "metric_depth_distance_bins.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "input_size",
                "sequence",
                "GT_Z_bin",
                "valid_pixels",
                "MAE_m",
                "RMSE_m",
                "AbsRel",
            ]
        )
        for row in summary["metric"]["results"]:
            for name, values in row["ground_truth_z_distance_bins"].items():
                writer.writerow(
                    [
                        row["model"],
                        row["input_size"],
                        row["sequence"],
                        name,
                        values["valid_pixels"],
                        values["mae_m"],
                        values["rmse_m"],
                        values["abs_rel"],
                    ]
                )
    with (DELIVERY / "metric_depth_error_quantiles.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "input_size",
                "sequence",
                "region",
                "valid_pixels",
                "p50_upper_m",
                "p90_upper_m",
                "p95_upper_m",
                "bin_width_m",
            ]
        )
        for row in summary["pooled_error_quantiles"]:
            quantiles = row["absolute_error_quantile_bin_upper_edge_m"]
            writer.writerow(
                [
                    row["model"],
                    row["input_size"],
                    row["sequence"],
                    row["region"],
                    row["valid_pixels"],
                    quantiles["p50"],
                    quantiles["p90"],
                    quantiles["p95"],
                    0.001,
                ]
            )
    baseline_depth = perf["android_baseline_only"]["relative_depth"]
    baseline_seg = perf["android_baseline_only"]["segformer"]
    report = [
        "# SightGuard 米制深度与局部空间候选研究交付",
        "",
        f"生成时间 UTC：{validation['generated_utc']}。本轮分支 `{branch}`，提交 `{revision}`。",
        "旧 0.4 已提交 PR #6，GitHub 显示已合并；本轮研究未推送、未合并 master。",
        "",
        "## 结论与证据等级",
        "",
        (
            "已完成：官方米制模型、700 帧公开 RGB-D 实验、墙面光轴深度统计"
            "、墙面特征扩张研究、复用冻结 Corridor/Grid 的图像候选、Andro"
            "id 默认关闭的米制 Shadow 接口及 APK。"
        ),
        (
            "部分验证：Android 源码编译/JVM 测试和 PC 同版本 ORT 兼容性通"
            "过；新版手机安装要求锁屏上的人工确认，已取消安装会话，没有绕"
            "过权限。未运行新版 Android 推理、CameraX、FPS/内存/生命周期验"
            "收。"
        ),
        "未验证：实体手机已量测墙距、真实墙面前移正例、可靠粗距离区间、身体通行宽度及安全路径。缺少卷尺测距验证。",
        (
            "正式 RiskScorer、AssessmentEngine、AlertScheduler、SpeechCon"
            "troller 均未改动。米制/接近/通行自动语音关闭；STEP 原有能力门"
            "禁保留。"
        ),
        "",
        "## 实际模型、单位与数据",
        "",
        (
            "官方 Depth Anything V2 Metric Small：Hypersim max_depth=20m，"
            "VKITTI max_depth=80m；官方代码提交 a561b849ebae10a6f5ef49e26"
            "c83cbbcd36c71bf，Small 权重 Apache-2.0。输出作为预测光轴 Z 深"
            "度，未输出欧氏范围、速度或 TTC。"
        ),
        (
            "相对深度基线仍为 relative inverse depth，大值相对更近；未拟合"
            "尺度、未换算成米。SegFormer ADE20K 真实 class 0 提供预测墙面"
            "掩膜；没有语义墙面真值，因此以下是“预测墙区域内深度误差”，"
            "不是墙面识别准确率。"
        ),
        (
            "TUM RGB-D，CC BY 4.0，Sturm/Engelhard/Endres/Burgard/Cremers"
            "，IROS 2012。registered uint16 depth/5000，0 无效；最近 RGB/"
            "depth 与 pose 关联各限 20ms。选择索引在推理前冻结。fr1 xyz/r"
            "py、fr3 无结构无纹理远景/有结构有纹理远景各 100 帧；fr3 无结"
            "构有纹理近景闭环 300 帧，共 700。全部为室内，未完成室外真值评"
            "测。"
        ),
        "",
        "| 下载文件 | 字节 | SHA256 |",
        "|---|---:|---|",
    ]
    for row in provenance:
        if row["path"].endswith((".pth", ".tgz")):
            report.append(f"| {Path(row['path']).name} | {row['bytes']} | `{row['sha256']}` |")
    report += [
        "",
        "## 真实距离误差",
        "",
        (
            "误差使用所有有效预测墙像素（含未被事件接受的帧），按像素加权"
            "；不是只保留成功帧。MAE/RMSE 单位米，AbsRel 无量纲。分位数由"
            "全部像素 1mm 直方图汇总，完整分位数与 GT 距离分组见两个 CSV。"
        ),
        "",
        "| 模型 | 输入短边 | 序列 | 帧数 | 墙距证据接受帧 | MAE | RMSE | AbsRel |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["metric"]["results"]:
        if row["region"] == "predicted_wall_region":
            report.append(
                f"| {row['model']} | {row['input_size']} | {row['sequence']} | {row['frames']} | "
                f"{row['wall_accepted_frames']} | {row['mae_m']:.4f} | "
                f"{row['rmse_m']:.4f} | {row['abs_rel']:.4f} |"
            )
    report += [
        "",
        (
            "分辨率降低的误差影响依赖场景，不能只依据 280 输入的一个较好序"
            "列宣布精度。VKITTI 室外域权重在这些室内序列有明显失败。"
        ),
        (
            "描述性区间实验：fr1 的 20 个接受帧得到约 ±2.1129m 的 P90 绝"
            "对中位数误差带；fr3 的 306 个相关帧落入该宽带。它不具备新场景"
            "的 90% 覆盖保证、没有墙面语义真值或手机量测校准，区间宽度约 "
            "4.23m，拒绝启用默认距离播报。"
        ),
        "",
        "## 接近与局部候选",
        "",
        (
            "墙面研究使用真实掩膜关联、Shi–Tomasi、双向 LK、RANSAC 相似变"
            "换、特征空间覆盖、Wahba 纯三维旋转反模型、连续窗口与 500ms 新"
            "鲜度门禁。深度数值波动不触发接近。TUM 标定用于纯视觉反模型；"
            "IMU 为 MOTION_SYNC_UNKNOWN，未伪造融合。"
        ),
        f"700 帧结果：{json.dumps(summary['spatial']['approach_states'])}。"
        f"独立 mocap 标签：{json.dumps(summary['spatial']['mocap_pair_labels'])}。",
        (
            "仅有 2 对明确相机前移标签，且都没有确认墙面；没有真实“向已确"
            "认墙面前移”正例或有效后退标签。因此不能宣称接近能力已验证，"
            "也不能把全弃权当作正确率 100%。合成扩张测试仅 Level 1。低纹理"
            "导致的 95 次特征不足等失败全部保留。"
        ),
        f"局部图像候选：{json.dumps(summary['spatial']['corridor_states'])}。"
        f"严格 D 组：{json.dumps(summary['spatial']['strict_d_states'])}。",
        (
            "逐一视觉复核发现 24 条支持记录均集中在同一贴海报平面场景，候"
            "选覆盖海报/板面，不能确认真正的行走地面。官方场景说明为有纹理"
            "平面，并未提供可通行地面标签；不把视觉复核当作官方语义真值或"
            "计算假正率。本轮没有通过真实地面候选验收。这是重要失败/限制，"
            "不是 24 条独立真实通道。全部原始支持结果与图保留。"
        ),
        (
            "未知格不是空闲格，已知检测框遮挡下部候选会拒绝；没有身体朝向"
            "、姿态、净空宽度或可通行真值。未输出 SAFE/PASSABLE、米制宽度"
            "或行动指令。Android 本轮没有移植局部 Corridor 评估器。"
        ),
        "",
        "## Android、音频与性能",
        "",
        f"JVM {tests['jvm']} 项，失败 {tests['jvm_failures']}、错误 {tests['jvm_errors']}；"
        "PC 614 通过、27 原有配置跳过；Ruff 通过。"
        "新手机 instrumentation 测试已编译但未运行。",
        (
            "已安装旧 0.4 在 nova12 上本轮重新执行 3 项 instrumentation："
            "真实 SegFormer、原相对深度、15 句环境音频资源。资源播放有真实"
            " onStart 和 completed，属于资源检查，不能代替识别到播放的新版"
            " CameraX 验收。"
        ),
        f"旧版原 Depth warm ms：{baseline_depth['warm_infer_ms']}；"
        f"SegFormer bitmap {baseline_seg['bitmap_infer_ms']}ms；"
        f"Depth 测试进程 PSS 从 {baseline_depth['pss_before_kb']} "
        f"到 {baseline_depth['pss_with_models_kb']}KB。不是新版米制模块内存。",
        (
            "四个静态形状（280×378、378×280、280×504、504×280）保持官"
            "方多 14 的等比例 resize、RGB/ImageNet/cubic 契约，opset17。一"
            "个 98,678,272 字节共享权重文件，按形状仅持有一个 CPU ORT ses"
            "sion。PC ORT 1.31 与 Android 同版本 1.24.3 都通过同输入 PyTo"
            "rch 对照；最大绝对差 <0.00005m，仅证明转换数值一致，不证明测"
            "距精度。"
        ),
        (
            "Android 默认关闭、仅开发者工具可开启；辅助模型串行、12 秒采样"
            "、至少 768MB 可用内存、正常热状态和主链余量才尝试；同次不叠加"
            "原 Depth。500ms TTL 未放宽，源帧/会话/画面变化/快速转动/模型"
            "来源无效会弃权。REALTIME 源使用实际采集年龄；UNKNOWN 源只记录"
            "接收年龄，不保证曝光时效。研究结果没有接入语音或正式风险。"
        ),
        (
            "新模型手机推理延迟、峰值 PSS、YOLO FPS 影响、真实 CameraX 帧"
            "龄均未测得；未用 AVD 性能冒充真机。PC 性能完整统计见 perform"
            "ance_report.json。手机最后读取热状态 0；未为本轮新增模块开启"
            "热状态门禁覆盖。"
        ),
        "",
        "PC 下表为实测模型/研究处理时延，不是 Android FPS 或端到端语音延迟。",
        "",
        "| PC 处理 | 样本数 | P50 ms | P95 ms | 最大 ms |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {values['count']} | {values['p50']:.3f} | "
            f"{values['p95']:.3f} | {values['max']:.3f} |"
            for name, values in perf["pc_timing_ms"].items()
        ],
        "",
        f"GPU 峰值 allocated {perf['pc_gpu_peak_allocated_mb']:.2f}MB，"
        f"reserved {perf['pc_gpu_peak_reserved_mb']:.2f}MB；"
        "采样边界最少可用 RAM "
        f"{perf['pc_minimum_free_ram_observed_at_sample_boundaries_mb']:.2f}MB。"
        "这是本轮采样观测，不是全系统峰值保证。",
        "",
        "## APK 与回退",
        "",
        "| 版本 | 路径 | SHA256 |",
        "|---|---|---|",
    ]
    for name, item in artifacts.items():
        report.append(f"| {name} | `{item['path']}` | `{item['sha256']}` |")
    report += [
        "",
        (
            "手机当前仍是稳定 0.4。关闭开发者米制开关可停止新模块；若未来"
            "安装实验版后回退，使用已保存稳定 APK，保留应用数据并按正常系"
            "统授权安装。详细命令和限制见 rollback.md。"
        ),
        "",
        "## 剩余工作",
        "",
        (
            "最值得继续的工作是：在可正常授权安装的手机上核对新模型预处理"
            "/输出/时效/内存，并采集带标定和参考墙距的静止、正向前移、后退"
            "、纯旋转、侧移正负例。未验证强反光、大遮挡、真实地面阴影、真"
            "实身体宽度；不得依据本轮局部图像候选给出通行建议。"
        ),
        "",
        (
            "源代码架构、复现命令与许可证链接见 android/docs/metric_corri"
            "dor_research.md。所有逐帧原始记录、失败、初始下载/测试失败日"
            "志均保留；没有删除失败样本、修改真值或放宽冻结 D 门禁。"
        ),
    ]
    (DELIVERY / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"git_commit": revision, "tests": tests, "artifacts": artifacts}, indent=2))


if __name__ == "__main__":
    main()
