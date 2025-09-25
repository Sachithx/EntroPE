# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel


class PlotEntropiesConfig(BaseModel):
    data_path: str | None
    chart_path: str
    score_override_path: str | None = None
    threshold_override: float | None = None

    class Config:
        extra = "forbid"


class PlotEntropiesData(BaseModel):
    text: str
    threshold: float = 1.335442066192627
    dataframe_json: str | None

    class Config:
        extra = "forbid"


def main():
    config_path = sys.argv[1]
    file_config = OmegaConf.load(config_path)
    # Omit program name and config file name
    cli_conf = OmegaConf.from_cli(sys.argv[2:])
    conf_dict = OmegaConf.to_container(
        OmegaConf.merge(file_config, cli_conf), resolve=True, throw_on_missing=True
    )
    plot_config = PlotEntropiesConfig(**conf_dict)
    with open(plot_config.data_path) as f:
        json_data = f.read()

    plot_data = PlotEntropiesData.model_validate_json(json_data)
    df = pd.read_json(plot_data.dataframe_json)
    print("LEN", len(df))
    if plot_config.threshold_override is None:
        threshold = plot_data.threshold
    else:
        threshold = plot_config.threshold_override
    if plot_config.score_override_path is not None:
        with open(plot_config.score_override_path) as f:
            scores = json.load(f)["score"]
            assert len(scores) == len(df)
            df["entropies"] = scores
            df["start"] = [1] + (df["entropies"] > threshold).values.tolist()[:-1]

    x_ticks = []
    for row in df.itertuples():
        position = row.position
        token = row.tokens
        x_ticks.append(f"{str(position).zfill(3)}|{token}")
    df["position_with_token"] = x_ticks
    print(df)

    x_axis = alt.Axis(
        labelExpr="split(datum.label, '|')[1]",
        grid=False,
        labelOverlap=False,
        labelAngle=0,
    )
    width = 1200
    height = 150
    base = alt.Chart(df).properties(width=width, height=height)
    points = base.mark_line(point=True).encode(
        x=alt.X("position_with_token:O", title=None, axis=x_axis),
        y=alt.Y(
            "entropies",
            title="Entropy of Next Byte",
        ),
    )
    rule = base.mark_rule(color="red", strokeDash=[4, 4]).encode(
        y=alt.datum(threshold),
    )
    patch_rules = (
        alt.Chart(df[df["start"] > 0])
        .properties(width=width, height=height)
        .mark_rule(color="#474747", strokeDash=[4, 2])
        .encode(x=alt.X("position_with_token:O", axis=x_axis))
    )

    chart = patch_rules + rule + points
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15)
    path = Path(plot_config.chart_path)
    path.parent.mkdir(exist_ok=True)
    chart.save(path)


if __name__ == "__main__":
    main()
