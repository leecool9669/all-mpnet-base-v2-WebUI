# -*- coding: utf-8 -*-
"""all-mpnet-base-v2 句子相似度 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：all-mpnet-base-v2 已就绪（演示模式，未加载真实权重）"


def fake_similarity(source: str, sentences: str) -> str:
    """模拟源句子与候选句子的相似度并返回可视化描述。"""
    if not (source or "").strip():
        return "请输入源句子。"
    lines_raw = [s.strip() for s in (sentences or "").strip().split("\n") if s.strip()]
    if not lines_raw:
        return "请输入至少一句候选句子，每行一句。"
    k = min(len(lines_raw), 10)
    lines = [
        "[演示] 已对句子相似度进行计算（未加载真实模型）。",
        f"源句子：{source[:80]}{'...' if len(source) > 80 else ''}",
        "",
        f"与 {k} 句候选的相似度示例（占位）：",
    ]
    for i in range(k):
        lines.append(f"  {i+1}. {lines_raw[i][:50]}{'...' if len(lines_raw[i]) > 50 else ''} -> 相似度: 0.{9 - i % 10}xx")
    lines.append("\n加载真实 all-mpnet-base-v2 后，将在此显示实际余弦相似度。")
    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="all-mpnet-base-v2 Sentence Similarity WebUI") as demo:
        gr.Markdown("## all-mpnet-base-v2 句子相似度 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 all-mpnet-base-v2 句子编码模型的典型使用流程，"
            "包括模型加载状态与句子相似度结果展示。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("句子相似度"):
                gr.Markdown(
                    "在下方输入源句子与多句候选（每行一句），模型将计算源句与各候选的语义相似度。"
                )
                source_inp = gr.Textbox(
                    label="源句子",
                    placeholder="例如：That is a happy person",
                    lines=2,
                )
                candidates_inp = gr.Textbox(
                    label="候选句子（每行一句）",
                    placeholder="例如：\nThat is a happy dog\nThat is a very happy person\nToday is a sunny day",
                    lines=6,
                )
                out = gr.Textbox(label="相似度结果说明", lines=12, interactive=False)
                run_btn = gr.Button("计算相似度（演示）")
                run_btn.click(
                    fn=fake_similarity,
                    inputs=[source_inp, candidates_inp],
                    outputs=out,
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 all-mpnet-base-v2 模型参数。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8762, share=False)


if __name__ == "__main__":
    main()
