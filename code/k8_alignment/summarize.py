"""Report all validation candidates and the committed test selection honestly."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import ROOT, RUN, NAMES, ds_for_run, sha256, save_json, metric


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |",
                      "|" + "|".join(["---"] * len(head)) + "|"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    results = json.loads((RUN / "results.json").read_text())
    selection = json.loads((RUN / "selection.json").read_text())
    assert results["selection_sha256"] == sha256(RUN / "selection.json")
    rows, intervals, summary = results["records"], [], {}
    labels = {"autolaparo": "AutoLaparo", "pedi38": "PediFundus38"}
    configs = {"dense16": "原Dense16", "baseline_k8": "原TS+ k8",
               "selected_k8": "验证集选出的TS+ k8"}
    control_checks = []
    for name in NAMES:
        for config, historical in (("dense16", "dense16"), ("baseline_k8", "ts8")):
            old_root = (ROOT / "outputs/autolaparo_unified_20261021/eval/adapt_e07/test"
                        if name == "autolaparo" else ROOT / "outputs/core9_20261021/eval/pedi38")
            for seed in range(3):
                new = RUN / "eval" / name / config / f"seed{seed}"
                old = old_root / historical / f"seed{seed}"
                a = json.loads((new / "metrics.json").read_text())
                b = json.loads((old / "metrics.json").read_text())
                for level in ("window", "frame"):
                    assert a[level] == b[level]
                for vid in ds_for_run(name).ids("test"):
                    old_file = f"{vid}.npz" if name == "autolaparo" else f"{vid}_draw0.npz"
                    with np.load(new / f"{vid}.npz") as aa, np.load(old / old_file) as bb:
                        assert np.array_equal(aa["prediction"], bb["prediction"])
                        assert np.array_equal(aa["selected_indices"], bb["selected_indices"])
                control_checks.append({"dataset": name, "config": config, "seed": seed,
                                       "metrics_and_all_predictions_match": True})
    for name in NAMES:
        ds = ds_for_run(name)
        n = len(ds.ids("test"))
        draws = np.random.default_rng(20261021 + n).integers(0, n, (2000, n))
        summary[name] = {}
        for config in configs:
            rr = sorted([r for r in rows if (r["dataset"], r["config"]) == (name, config)],
                        key=lambda r: r["seed"])
            assert [r["seed"] for r in rr] == [0, 1, 2]
            j = [r["window"]["jaccard"] for r in rr]
            summary[name][config] = {
                "j": float(np.mean(j)), "sd": float(np.std(j)), "seeds": j,
                "f1": float(np.mean([r["window"]["f1"] for r in rr])),
                "accuracy": float(np.mean([r["window"]["accuracy"] for r in rr])),
                "frame_j": float(np.mean([r["frame"]["jaccard"] for r in rr]))}
        for baseline in ("dense16", "baseline_k8"):
            bootstrap, points = [], []
            for seed in range(3):
                a_path = RUN / "eval" / name / "selected_k8" / f"seed{seed}/confusions.npz"
                b_path = RUN / "eval" / name / baseline / f"seed{seed}/confusions.npz"
                with np.load(a_path) as aa, np.load(b_path) as bb:
                    assert aa["videos"].tolist() == bb["videos"].tolist() == ds.ids("test")
                    a, b = aa["window"], bb["window"]
                bootstrap.append(metric(a[draws].sum(1))["jaccard"] -
                                 metric(b[draws].sum(1))["jaccard"])
                points.append(metric(a.sum(0))["jaccard"] - metric(b.sum(0))["jaccard"])
            intervals.append({"dataset": name, "comparison": f"selected_k8-minus-{baseline}",
                              "difference": float(np.mean(points)),
                              "ci95": np.quantile(np.mean(bootstrap, axis=0), [.025, .975]).tolist()})
    save_json(RUN / "bootstrap.json", {"iterations": 2000, "records": intervals})
    save_json(RUN / "summary.json", {"state": "complete", "datasets": summary})
    out = ["## 19. k8训练对齐补实验：按验证集选择，不保证测试集上升",
           "本轮固定AutoLaparo/Pedi38已有划分、打分器、ViT、16帧窗口和k8推理预算，"
           "从原k8 BiGRU继续训练。原Dense16不重训、不替换；原核心9配置仍保留在第2–10节。"
           "以下是额外调优的探索性结果，不与之前的耗时组成一套已重新测量的结果。",
           "既有k8 BiGRU按分bin随机采样训练，checkpoint也按随机采样验证J选择。"
           "本轮改为实际TS+ k8验证J选checkpoint，比较两种预先固定的训练配方："
           "native使用实际TS+8输入和原加权交叉熵；mixed_distill按窗口以各50%的概率使用TS+8或随机8帧，"
           "并加入同种子Dense16教师的软预测监督（温度2，KL系数0.5）。"
           "教师只处理训练视频，推理没有教师，仍只编码8帧。",
           "两数据集均为2配方×3种子，每项20epoch，Adam学习率1e-4、weight decay1e-4。"
           "每种子纳入原checkpoint作为epoch0，取验证J最高的最早epoch；"
           "再按三种子平均验证J选择数据集的配方，平局优先原模型。"
           "两个数据集的最终checkpoint及哈希均写入selection.json后，才运行本轮测试集评估。"]
    val_rows = []
    for name in NAMES:
        selected = selection["datasets"][name]
        for recipe in ("baseline", "native", "mixed_distill"):
            record = json.loads((RUN / f"validation_{name}.json").read_text())
            epochs = "/".join(str(r["epoch"]) for r in record["candidates"][recipe])
            decision = ("选中" if recipe == selected["recipe"] else
                        "参照；重算测试核对" if recipe == "baseline" else "未选；不补测测试集")
            val_rows.append([labels[name], recipe, f"{selected['mean_val_j'][recipe]:.6f}", epochs, decision])
    out += ["### 19.1 全部验证候选",
            table(["数据集", "配方", "三种子平均验证J", "seed0/1/2所选epoch", "决策"], val_rows)]
    test_rows, seed_rows = [], []
    for name in NAMES:
        dense_j = summary[name]["dense16"]["j"]
        for config, title in configs.items():
            v = summary[name][config]
            test_rows.append([labels[name], title, 16 if config == "dense16" else 8,
                              f"{v['j']:.6f} ± {v['sd']:.6f}", f"{100*(v['j']-dense_j):+.3f}",
                              f"{v['f1']:.6f}", f"{v['accuracy']:.6f}", f"{v['frame_j']:.6f}"])
            seed_rows.append([labels[name], title, *[f"{j:.6f}" for j in v["seeds"]]])
    out += ["### 19.2 测试结果",
            table(["数据集", "配置", "编码帧数", "窗口J均值±SD", "相对原Dense ΔJ(pp)",
                   "窗口F1", "窗口Accuracy", "帧展开J"], test_rows),
            table(["数据集", "配置", "seed0 J", "seed1 J", "seed2 J"], seed_rows),
            "均值和总体SD均来自三个既有下游种子的延续训练，打分器固定。"
            "测试集指标采用完整固定类别集合；未删除类别、视频或不利种子。"
            "帧展开结果由同一窗口预测展开，不是独立证据。"
            "12项原Dense/k8控制评估的全部测试视频预测、选帧索引和各项指标均与历史结果完全一致。"]
    out += [table(["数据集", "比较", "ΔJ(pp)", "成对视频bootstrap 95%区间(pp)"],
                  [[labels[r["dataset"]], r["comparison"], f"{100*r['difference']:+.3f}",
                    f"[{100*r['ci95'][0]:+.3f}, {100*r['ci95'][1]:+.3f}]"] for r in intervals])]
    findings = []
    for name in NAMES:
        delta = 100 * (summary[name]["selected_k8"]["j"] - summary[name]["dense16"]["j"])
        change = 100 * (summary[name]["selected_k8"]["j"] - summary[name]["baseline_k8"]["j"])
        findings.append(f"{labels[name]}选择`{selection['datasets'][name]['recipe']}`，"
                        f"测试J相对原Dense为{delta:+.3f}pp，相对原k8为{change:+.3f}pp。")
    out += ["### 19.3 结论与适用范围", " ".join(findings),
            "区间来自2000次成对视频重采样，条件于当前模型和划分，未作多重比较校正。"
            "点估计转正不自动等于稳定或显著优于Dense；只有本轮结果实际支持时才表述为上升。",
            "这批测试数据在此前工作中已多次查看，因此本轮即使按验证集选模型，仍属于探索性改进。"
            "TS+获得额外训练和验证搜索，Dense保持原结果，调优预算不对等；不能将差值全部归因于选帧方法。"
            "没有根据本轮测试结果继续换配方、选seed或回退成测试分数更高的候选。",
            "模型结构、k和视觉输入不变，所以编码帧数和理论视觉FLOPs不因本次权重更新而改变。"
            "本轮只重新评估缓存特征上的精度，没有重新测量文件读取至预测的完整管线耗时/显存；"
            "新精度不能标为与旧耗时同场重测。后续若用于最终主表，需要补齐新权重的图像管线核对与计时。",
            f"实验协议：[protocol.json]({RUN / 'protocol.json'})；"
            f"测试前选择：[selection.json]({RUN / 'selection.json'})；"
            f"逐种子结果：[results.json]({RUN / 'results.json'})；"
            f"区间：[bootstrap.json]({RUN / 'bootstrap.json'})。"]
    text = "\n\n".join(out) + "\n"
    (RUN / "补实验报告.md").write_text(text)
    archived = RUN / "source_final"
    archived.mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, archived / source.name)
    save_json(RUN / "audit.json", {
        "state": "complete", "result_count": len(rows), "selection_hash_verified": True,
        "historical_control_checks": control_checks,
        "all_three_seeds": True, "test_videos": {name: ds_for_run(name).ids("test") for name in NAMES},
        "files": {str(p.relative_to(RUN)): sha256(p) for p in RUN.rglob("*")
                  if p.is_file() and p.name != "audit.json" and "scores" not in p.parts}})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
