"""Tests for dynamic ONNX model export."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator

from transitlens_ml_core.export import export_onnx

from .conftest import configured_model, export_config


def test_onnx_export_is_valid_dynamic_and_self_describing(tmp_path: Path) -> None:
    config = export_config(tmp_path / "nested")
    model = configured_model(config)
    model.train()

    path = export_onnx(model, config)
    exported = onnx.load(path)
    onnx.checker.check_model(exported)
    metadata = {item.key: item.value for item in exported.metadata_props}

    assert model.training
    assert path == config.export.output_directory / config.export.onnx_filename
    assert exported.opset_import[0].version == config.export.onnx_opset_version
    assert exported.graph.input[0].name == "light_curve"
    assert exported.graph.output[0].name == "transit_probability"
    input_shape = exported.graph.input[0].type.tensor_type.shape.dim
    output_shape = exported.graph.output[0].type.tensor_type.shape.dim
    assert input_shape[0].dim_param == "batch"
    assert input_shape[1].dim_value == 1
    assert input_shape[2].dim_param == "samples"
    assert output_shape[0].dim_param == "batch"
    assert metadata["transitlens.schema_version"] == "1"
    assert metadata["transitlens.model_version"] == config.project.version
    assert metadata["transitlens.input_field"] == config.data.model_input_field
    assert json.loads(metadata["transitlens.model_config"]) == config.model.model_dump(
        mode="json"
    )


@pytest.mark.parametrize(("batch_size", "sample_length"), [(1, 16), (3, 32), (2, 65)])
def test_onnx_output_matches_pytorch_for_dynamic_lengths(
    tmp_path: Path, batch_size: int, sample_length: int
) -> None:
    config = export_config(tmp_path)
    model = configured_model(config)
    path = export_onnx(model, config)
    evaluator = ReferenceEvaluator(onnx.load(path))
    sequence = np.linspace(0.9, 1.1, sample_length, dtype=np.float32).reshape(1, 1, -1)
    inputs = np.repeat(sequence, batch_size, axis=0)

    onnx_output = evaluator.run(None, {"light_curve": inputs})[0]
    with torch.inference_mode():
        torch_output = model(torch.from_numpy(inputs)).numpy()

    np.testing.assert_allclose(onnx_output, torch_output, rtol=1e-5, atol=1e-6)


def test_onnx_export_rejects_incompatible_model(tmp_path: Path) -> None:
    config = export_config(tmp_path)
    incompatible = configured_model(config)
    incompatible.classifier.output = torch.nn.Linear(16, 1)

    with pytest.raises(ValueError, match="incompatible"):
        export_onnx(incompatible, config)
