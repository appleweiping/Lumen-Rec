from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from main_train_score_setrec_upstream_adapter import (
    _enable_setrec_qwen_memory_saving,
    _import_official_model,
    _make_training_arguments,
    _patch_qwen2_decoder_layer_compat,
    _qwen2_attention_forward_compat,
    _qwen2_rotary_from_layer,
)

from src.baselines.official_runner.contract import resolve_method_config
from src.baselines.official_runner.setrec import run_setrec_official

PINNED_SETREC_COMMIT = "2ed9a75ad1ad3784c61bba3c68cbedbe3cfce2d7"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    _write_csv(
        task_dir / "train_interactions.csv",
        [
            {"user_id": "u1", "item_id": "i1", "timestamp": 1, "sequence_index": 1},
            {"user_id": "u1", "item_id": "i2", "timestamp": 2, "sequence_index": 2},
            {"user_id": "u2", "item_id": "i2", "timestamp": 1, "sequence_index": 1},
            {"user_id": "u2", "item_id": "i3", "timestamp": 2, "sequence_index": 2},
        ],
        ["user_id", "item_id", "timestamp", "sequence_index"],
    )
    _write_csv(
        task_dir / "candidate_items.csv",
        [
            {"source_event_id": "e1", "user_id": "u1", "item_id": "i2", "candidate_index": 0, "candidate_title": "Two"},
            {"source_event_id": "e1", "user_id": "u1", "item_id": "i3", "candidate_index": 1, "candidate_title": "Three"},
        ],
        ["source_event_id", "user_id", "item_id", "candidate_index", "candidate_title"],
    )
    (task_dir / "ranking_test.jsonl").write_text(json.dumps({"source_event_id": "e1"}) + "\n", encoding="utf-8")
    return task_dir


def _make_repo(tmp_path: Path) -> Path:
    repo_dir = tmp_path / "SETRec"
    for rel_path in [
        "code/finetune_qwen.py",
        "code/model_qwen.py",
        "code/Q_qwen.py",
        "code/parse_utils.py",
        "code/utils/data_utils.py",
        "code/AE/models/ae.py",
        "code/AE/models/layers.py",
    ]:
        path = repo_dir / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n", encoding="utf-8")
    return repo_dir


def _args(tmp_path: Path, *, task_dir: Path, repo_dir: Path, dry_run: bool = False) -> argparse.Namespace:
    output_dir = tmp_path / "official"
    return argparse.Namespace(
        method="setrec",
        stage="run",
        domain="beauty",
        task_dir=str(task_dir),
        valid_task_dir="",
        output_scores_path=str(output_dir / "setrec_official_scores.csv"),
        provenance_output_path=str(output_dir / "fairness_provenance.json"),
        fairness_policy_id="official_code_qwen3base_default_hparams_declared_adaptation_v1",
        comparison_variant="official_code_qwen3base_default_hparams_declared_adaptation",
        backbone_model_family="Qwen3-8B",
        backbone_path=str(tmp_path / "Qwen3-8B"),
        llm_adaptation_mode="official_adapter",
        implementation_status="",
        repo_dir=str(repo_dir),
        work_dir="",
        adapter_or_checkpoint_path="",
        hparam_policy="official_default_or_recommended",
        baseline_hyperparameter_overrides_json="{}",
        validation_selection_metric="none",
        run_id="",
        seed=42,
        device="cpu",
        output_root=str(tmp_path / "outputs"),
        dry_run=dry_run,
        force_embeddings=False,
        embedding_backend="hf_mean_pool",
        embedding_batch_size=2,
        embedding_max_text_chars=1200,
        embedding_max_length=16,
        torch_dtype="auto",
        hf_device_map="",
        trust_remote_code=False,
        deterministic_dim=384,
        score_batch_size=128,
        setrec_adapter_exp_name="",
        setrec_adapter_dir="",
        setrec_epochs=20,
        setrec_train_batch_size=512,
        setrec_micro_batch_size=1,
        setrec_lr=3.0e-4,
        setrec_max_len=50,
        setrec_val_set_size=2000,
        setrec_n_sem=1,
        setrec_n_cf=1,
        setrec_alpha=0.7,
        setrec_beta=0.1,
        setrec_ae_layers="512,256,128",
        setrec_dropout_prob=0.0,
        setrec_bn=False,
        setrec_loss_type="mse",
        setrec_sem_encoder="qwen",
        setrec_lora_r=8,
        setrec_lora_alpha=16,
        setrec_lora_dropout=0.02,
        setrec_lora_target_modules="q_proj,v_proj,o_proj",
        setrec_warmup_steps=100,
        setrec_lr_scheduler="cosine",
    )


def _config() -> tuple[dict, dict, dict]:
    return resolve_method_config("configs/official_external_baselines.yaml", "setrec")


def test_setrec_official_runner_marks_completed_only_after_exact_scores(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path)
    repo_dir = _make_repo(tmp_path)
    (tmp_path / "Qwen3-8B").mkdir()
    args = _args(tmp_path, task_dir=task_dir, repo_dir=repo_dir)
    cfg, method_cfg, contract = _config()

    monkeypatch.setattr("src.baselines.official_runner.setrec.git_commit", lambda _repo: PINNED_SETREC_COMMIT)

    def fake_embeddings(adapter_dir: Path, **_: object) -> dict[str, str]:
        handled = adapter_dir / "llm_esr" / "handled"
        with (handled / "itm_emb_np.pkl").open("wb") as fh:
            pickle.dump(np.ones((3, 8), dtype=np.float32), fh)
        with (handled / "pca64_itm_emb_np.pkl").open("wb") as fh:
            pickle.dump(np.ones((3, 4), dtype=np.float32), fh)
        (adapter_dir / "llmesr_embedding_metadata.json").write_text(
            json.dumps({"backend": "hf_mean_pool", "artifact_class": "adapter_text_embedding"}) + "\n",
            encoding="utf-8",
        )
        return {
            "itm_emb_path": str(handled / "itm_emb_np.pkl"),
            "pca64_emb_path": str(handled / "pca64_itm_emb_np.pkl"),
        }

    def fake_train(train_args: argparse.Namespace) -> dict[str, object]:
        candidate_rows = list(csv.DictReader((task_dir / "candidate_items.csv").open(newline="", encoding="utf-8")))
        _write_csv(
            Path(train_args.output_scores_path),
            [
                {"source_event_id": row["source_event_id"], "user_id": row["user_id"], "item_id": row["item_id"], "score": 1.0 - idx}
                for idx, row in enumerate(candidate_rows)
            ],
            ["source_event_id", "user_id", "item_id", "score"],
        )
        Path(train_args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(train_args.checkpoint_path).write_bytes(b"checkpoint")
        return {
            "status": "completed",
            "checkpoint_path": train_args.checkpoint_path,
            "official_data_dir": str(tmp_path / "SETRec" / "data" / "beauty"),
            "trainable_params": 12,
        }

    monkeypatch.setattr("src.baselines.official_runner.setrec.generate_sentence_embeddings", fake_embeddings)
    monkeypatch.setattr("src.baselines.official_runner.setrec.train_and_score", fake_train)

    provenance = run_setrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)

    assert provenance["implementation_status"] == "official_completed"
    assert provenance["blockers"] == []
    assert provenance["score_coverage_rate"] == 1.0
    assert provenance["runner_support_level"] == "official_setrec_qwen3base_identifier"
    assert provenance["official_training_config"]["model_name"] == "Qwen4Rec"
    assert provenance["official_training_config"]["batch_size"] == 512
    assert provenance["official_training_config"]["micro_batch_size"] == 1
    assert provenance["official_training_config"]["n_sem"] == 1
    assert provenance["official_training_config"]["gradient_accumulation_steps"] == 512
    assert provenance["official_training_config"]["effective_batch_size"] == 512
    assert provenance["baseline_hyperparameter_overrides"]["micro_batch_size"]["official_default"] == 64
    assert provenance["baseline_hyperparameter_overrides"]["gradient_checkpointing"]["runner_value"] is True
    assert provenance["baseline_hyperparameter_overrides"]["use_cache"]["runner_value"] is False
    assert provenance["same_candidate_score_audit"]["audit_ok"] is True


def test_setrec_official_runner_blocks_hash_embeddings(tmp_path, monkeypatch):
    task_dir = _make_task(tmp_path)
    repo_dir = _make_repo(tmp_path)
    (tmp_path / "Qwen3-8B").mkdir()
    args = _args(tmp_path, task_dir=task_dir, repo_dir=repo_dir, dry_run=True)
    args.embedding_backend = "deterministic_text_hash"
    cfg, method_cfg, contract = _config()

    monkeypatch.setattr("src.baselines.official_runner.setrec.git_commit", lambda _repo: PINNED_SETREC_COMMIT)

    provenance = run_setrec_official(args=args, cfg=cfg, method_cfg=method_cfg, contract=contract)

    assert provenance["implementation_status"] == "official_blocked"
    assert "setrec_official_requires_real_text_embeddings_not_deterministic_hash" in provenance["blockers"]
    assert provenance["score_coverage_rate"] is None


def test_setrec_official_import_patches_qwen_extra_init_kwargs(tmp_path, monkeypatch):
    repo_dir = _make_repo(tmp_path)
    code_dir = repo_dir / "code"
    (code_dir / "Q_qwen.py").write_text(
        "class QQwen2Model:\n"
        "    def __init__(self, config):\n"
        "        self.torch_name = torch.__name__\n"
        "        self.nn_name = nn.__name__\n"
        "        self.config = config\n",
        encoding="utf-8",
    )
    (code_dir / "model_qwen.py").write_text(
        "from Q_qwen import QQwen2Model\n"
        "class Qwen4Rec:\n"
        "    pass\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(code_dir))

    model_cls = _import_official_model(repo_dir)
    from Q_qwen import QQwen2Model

    instance = QQwen2Model("config", load_in_8bit=True, device_map="auto")
    assert model_cls.__name__ == "Qwen4Rec"
    assert instance.config == "config"
    assert instance.torch_name == "torch"
    assert instance.nn_name == "torch.nn"


def test_setrec_training_arguments_adapts_eval_strategy_name(tmp_path):
    captured: dict[str, object] = {}

    class FakeTrainingArguments:
        def __init__(self, output_dir: str, eval_strategy: str = "", per_device_train_batch_size: int = 1) -> None:
            captured.update(
                {
                    "output_dir": output_dir,
                    "eval_strategy": eval_strategy,
                    "per_device_train_batch_size": per_device_train_batch_size,
                }
            )

    class FakeTransformers:
        TrainingArguments = FakeTrainingArguments

    args = argparse.Namespace(
        micro_batch_size=7,
        warmup_steps=100,
        epochs=20,
        lr=3.0e-4,
        lr_scheduler="cosine",
    )

    _make_training_arguments(
        transformers=FakeTransformers,
        args=args,
        checkpoint_dir=tmp_path / "ckpt",
        gradient_accumulation_steps=3,
    )

    assert captured["eval_strategy"] == "no"
    assert captured["per_device_train_batch_size"] == 7


def test_setrec_memory_bridge_enables_checkpointing_and_disables_cache():
    class FakeConfig:
        use_cache = True

    class FakeInner:
        config = FakeConfig()
        gradient_checkpointing = False

        def gradient_checkpointing_enable(self):
            self.gradient_checkpointing = True

        def enable_input_require_grads(self):
            self.input_grads = True

    class FakeBase:
        def __init__(self):
            self.model = FakeInner()

        def gradient_checkpointing_enable(self):
            self.enabled = True

        def enable_input_require_grads(self):
            self.input_grads = True

    class FakeQwen:
        def __init__(self):
            self.base_model = FakeBase()

    class FakeSetrec:
        def __init__(self):
            self.qwen_model = FakeQwen()

    summary = _enable_setrec_qwen_memory_saving(FakeSetrec())

    assert summary["qwen_use_cache"] is False
    assert summary["qwen_gradient_checkpointing"] is True


def test_setrec_rotary_compat_uses_actual_attention_head_dim():
    import torch

    class FakeAttention:
        head_dim = 32
        config = type("Config", (), {"rope_theta": 10000.0})()

    class FakeLayer:
        self_attn = FakeAttention()

    hidden_states = torch.zeros((2, 5, 64), dtype=torch.float16)
    position_ids = torch.arange(5).unsqueeze(0).repeat(2, 1)

    cos, sin = _qwen2_rotary_from_layer(layer=FakeLayer(), hidden_states=hidden_states, position_ids=position_ids)

    assert cos.shape == (2, 5, 32)
    assert sin.shape == (2, 5, 32)
    assert cos.dtype == torch.float16


def test_setrec_decoder_layer_wrapper_preserves_old_tuple_contract(monkeypatch):
    import torch
    Qwen2DecoderLayer = pytest.importorskip("transformers.models.qwen2.modeling_qwen2").Qwen2DecoderLayer

    original = Qwen2DecoderLayer.forward

    def fake_forward(self, hidden_states, *args, **kwargs):
        assert hidden_states.ndim == 3
        assert kwargs["position_embeddings"][0].shape[-1] == 32
        return hidden_states + 1

    monkeypatch.setattr(Qwen2DecoderLayer, "_pony_accepts_missing_position_embeddings", False, raising=False)
    monkeypatch.setattr(Qwen2DecoderLayer, "forward", fake_forward)

    _patch_qwen2_decoder_layer_compat()
    wrapped = Qwen2DecoderLayer.forward

    class FakeAttention:
        head_dim = 32
        config = type("Config", (), {"rope_theta": 10000.0})()

    class FakeLayer:
        self_attn = FakeAttention()

    hidden = torch.zeros((2, 1, 5, 64), dtype=torch.float32)
    position_ids = torch.arange(5).unsqueeze(0).repeat(2, 1)
    out = wrapped(FakeLayer(), hidden, position_ids=position_ids)

    assert isinstance(out, tuple)
    assert out[0].shape == (2, 5, 64)

    monkeypatch.setattr(Qwen2DecoderLayer, "forward", original)


def test_setrec_decoder_layer_wrapper_flattens_higher_rank_hidden(monkeypatch):
    import torch
    Qwen2DecoderLayer = pytest.importorskip("transformers.models.qwen2.modeling_qwen2").Qwen2DecoderLayer

    original = Qwen2DecoderLayer.forward

    def fake_forward(self, hidden_states, *args, **kwargs):
        assert hidden_states.shape == (2, 15, 64)
        assert kwargs["position_embeddings"][0].shape == (2, 15, 32)
        return hidden_states + 1

    monkeypatch.setattr(Qwen2DecoderLayer, "_pony_accepts_missing_position_embeddings", False, raising=False)
    monkeypatch.setattr(Qwen2DecoderLayer, "forward", fake_forward)

    _patch_qwen2_decoder_layer_compat()
    wrapped = Qwen2DecoderLayer.forward

    class FakeAttention:
        head_dim = 32
        config = type("Config", (), {"rope_theta": 10000.0})()

    class FakeLayer:
        self_attn = FakeAttention()

    hidden = torch.zeros((2, 1, 3, 5, 64), dtype=torch.float32)
    position_ids = torch.arange(5).unsqueeze(0).repeat(2, 1)
    out = wrapped(FakeLayer(), hidden, position_ids=position_ids)

    assert isinstance(out, tuple)
    assert out[0].shape == (2, 15, 64)

    monkeypatch.setattr(Qwen2DecoderLayer, "forward", original)


def test_setrec_decoder_layer_wrapper_normalizes_legacy_checkpoint_args(monkeypatch):
    import torch
    Qwen2DecoderLayer = pytest.importorskip("transformers.models.qwen2.modeling_qwen2").Qwen2DecoderLayer

    original = Qwen2DecoderLayer.forward

    def fake_forward(self, hidden_states, **kwargs):
        assert hidden_states.shape == (2, 5, 64)
        assert "past_key_value" not in kwargs
        assert kwargs["attention_mask"].shape == (2, 5)
        assert kwargs["position_ids"].shape == (2, 5)
        assert kwargs["past_key_values"] == "cache"
        assert kwargs["output_attentions"] is False
        assert kwargs["use_cache"] is False
        assert kwargs["position_embeddings"][0].shape == (2, 5, 32)
        return hidden_states + 1

    monkeypatch.setattr(Qwen2DecoderLayer, "_pony_accepts_missing_position_embeddings", False, raising=False)
    monkeypatch.setattr(Qwen2DecoderLayer, "forward", fake_forward)

    _patch_qwen2_decoder_layer_compat()
    wrapped = Qwen2DecoderLayer.forward

    class FakeAttention:
        head_dim = 32
        config = type("Config", (), {"rope_theta": 10000.0})()

    class FakeLayer:
        self_attn = FakeAttention()

    hidden = torch.zeros((2, 5, 64), dtype=torch.float32)
    mask = torch.ones((2, 5), dtype=torch.float32)
    position_ids = torch.arange(5).unsqueeze(0).repeat(2, 1)
    out = wrapped(FakeLayer(), hidden, mask, position_ids, "cache", False, False)

    assert isinstance(out, tuple)
    assert out[0].shape == (2, 5, 64)

    monkeypatch.setattr(Qwen2DecoderLayer, "forward", original)


def test_setrec_attention_wrapper_flattens_direct_attention_path(monkeypatch):
    import torch
    Qwen2Attention = pytest.importorskip("transformers.models.qwen2.modeling_qwen2").Qwen2Attention

    original = Qwen2Attention.forward

    def fake_attention_forward(self, hidden_states, position_embeddings, attention_mask=None, *args, **kwargs):
        assert hidden_states.shape == (2, 15, 64)
        assert position_embeddings[0].shape == (2, 15, 32)
        return hidden_states + 1, None

    monkeypatch.setattr(Qwen2Attention, "_pony_accepts_setrec_hidden_rank", False, raising=False)
    monkeypatch.setattr(Qwen2Attention, "forward", fake_attention_forward)

    _patch_qwen2_decoder_layer_compat()
    wrapped = Qwen2Attention.forward

    class FakeAttention:
        q_proj = lambda self, value: value
        k_proj = lambda self, value: value
        v_proj = lambda self, value: value
        o_proj = lambda self, value: value
        head_dim = 32
        num_key_value_groups = 1
        attention_dropout = 0.0
        training = False
        scaling = 32**-0.5
        sliding_window = None
        config = type("Config", (), {"rope_theta": 10000.0})()

    hidden = torch.zeros((2, 3, 5, 64), dtype=torch.float32)
    position_ids = torch.arange(5).unsqueeze(0).repeat(2, 1)
    out, weights = wrapped(FakeAttention(), hidden, position_embeddings=None, attention_mask=None, position_ids=position_ids)

    assert out.shape == (2, 15, 64)
    assert weights is None

    monkeypatch.setattr(Qwen2Attention, "forward", original)


def test_setrec_attention_compat_flattens_high_rank_lora_projection():
    import torch
    pytest.importorskip("transformers.models.qwen2.modeling_qwen2")

    class FakeProjection:
        def __init__(self, heads: int, head_dim: int, *, high_rank: bool = True) -> None:
            self.heads = heads
            self.head_dim = head_dim
            self.high_rank = high_rank

        def __call__(self, hidden_states):
            batch_size, seq_len, _hidden = hidden_states.shape
            projected = torch.zeros((batch_size, seq_len, self.heads * self.head_dim), dtype=hidden_states.dtype)
            if self.high_rank:
                return projected.reshape(batch_size, 1, seq_len, self.heads * self.head_dim)
            return projected

    class FakeOutProjection:
        def __call__(self, value):
            return value

    class FakeAttention:
        q_proj = FakeProjection(heads=2, head_dim=32)
        k_proj = FakeProjection(heads=2, head_dim=32)
        v_proj = FakeProjection(heads=2, head_dim=32)
        o_proj = FakeOutProjection()
        head_dim = 32
        attention_dropout = 0.0
        training = False
        scaling = 32**-0.5
        sliding_window = None

    hidden = torch.zeros((2, 15, 64), dtype=torch.float32)
    cos = torch.ones((2, 15, 32), dtype=torch.float32)
    sin = torch.zeros((2, 15, 32), dtype=torch.float32)

    output, weights = _qwen2_attention_forward_compat(
        attention=FakeAttention(),
        hidden_states=hidden,
        position_embeddings=(cos, sin),
        attention_mask=None,
    )

    assert output.shape == (2, 15, 64)
    assert weights is None
