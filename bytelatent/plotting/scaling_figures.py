# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import pydantic
from omegaconf import OmegaConf


class ScalingPlotsConfig(pydantic.BaseModel):
    df_dir: str
    output_chart_dir: str
    frame_files: list[str]

    class Config:
        extra = "forbid"


def determine_family(key: str):
    if key.startswith("Megabyte++"):
        return "Megabyte++"
    elif key.startswith("BLT"):
        return "BLT"
    elif key.startswith("LLaMA"):
        return "LLaMA"
    elif key.startswith("Space"):
        return "Space"


file_to_vars = {}


def create_chart(df: pd.DataFrame, output_file: str):
    df["metric"] = df["bpb/not_heldout.jsonl"]
    df["family"] = df["key"].map(determine_family)
    model_domain = [
        "BLT Space ps=6",
        "BLT Space w/o cross-attn",
        "SpaceByte",
        "LLaMA 3 BPE",
        "Megabyte++ ps=4",
        "Megabyte++ ps=6",
    ]
    color_range = ["#1f77b4", "#1f77b4", "#1f77b4", "#ff7f0e", "#2ca02c", "#2ca02c"]
    shape_range = [
        "circle",
        "square",
        "cross",
        "diamond",
        "triangle-up",
        "triangle-down",
    ]
    color_scale = alt.Scale(domain=model_domain, range=color_range)
    shape_scale = alt.Scale(
        domain=model_domain,
        range=shape_range,
    )
    base_chart = alt.Chart(df).encode(
        x=alt.X("flops", title="Training FLOPS")
        .scale(type="log", domain=[2e20, 1.25e22])
        .axis(values=[2e20, 4e20, 8e20, 1e21, 2e21, 4e21, 8e21, 1e22]),
        y=alt.Y("metric", title="Bits per Byte (BPB)").scale(zero=False),
    )
    lines = base_chart.encode(
        color=alt.Color("key", title="Model Color", scale=color_scale, legend=None),
        strokeDash=alt.StrokeDash("family", title="Model Family", legend=None),
    ).mark_line()
    points = base_chart.encode(
        color=alt.Color("key", title="Model", scale=color_scale),
        shape=alt.Shape("key", title="", scale=shape_scale),
    ).mark_point(size=70)
    chart = (
        (lines + points)
        .resolve_scale(
            color="independent",
            shape="independent",
            # strokeDash="independent",
        )
        .configure_legend(orient="right")
        .properties(height=300, width=400)
    )
    print("Saving", output_file)
    chart.save(output_file)


def main():
    config_path = sys.argv[1]
    file_config = OmegaConf.load(config_path)
    # Omit program name and config file name
    cli_conf = OmegaConf.from_cli(sys.argv[2:])
    conf_dict = OmegaConf.to_container(
        OmegaConf.merge(file_config, cli_conf), resolve=True, throw_on_missing=True
    )
    plot_config = ScalingPlotsConfig(**conf_dict)
    df_dir = Path(plot_config.df_dir)
    chart_dir = Path(plot_config.output_chart_dir)
    chart_dir.mkdir(exist_ok=True, parents=True)
    for ff in plot_config.frame_files:
        path = df_dir / ff
        df = pd.read_json(path)
        print(df)
        print(df.columns)
        create_chart(df, chart_dir / f"{path.name}.pdf")


if __name__ == "__main__":
    main()
