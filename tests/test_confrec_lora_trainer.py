"""The last-token trainer's loss equals the shifted full-sequence loss (prompt masked), and a LoRA run with it updates
the weights exactly as the stock Trainer does (gradient accumulation included). Needs transformers + torch (+ peft for
the trainer test): the GPU server's `lumen` env; skipped elsewhere."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.confrec.lora_trainer import last_token_loss, last_token_trainer  # noqa: E402
from src.confrec.train_lora_yesno import collate  # noqa: E402


def tiny_model():
    cfg = transformers.Qwen3Config(vocab_size=300, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=128)
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def batch(lengths, vocab=300, pad=0, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = max(lengths)
    ids = torch.full((len(lengths), m), pad, dtype=torch.long)
    lab = torch.full((len(lengths), m), -100, dtype=torch.long)
    att = torch.zeros((len(lengths), m), dtype=torch.long)
    for k, n in enumerate(lengths):                       # left padding, answer = last token (as train_lora_yesno.collate)
        seq = torch.randint(5, vocab, (n,), generator=g)
        ids[k, m - n:], att[k, m - n:] = seq, 1
        lab[k, -1] = seq[-1]
    return ids, att, lab


def test_matches_full_sequence_loss_with_left_padding():
    model = tiny_model()
    ids, att, lab = batch([12, 7, 9, 12])
    with torch.no_grad():
        ours = last_token_loss(model, ids, att, lab)
        full = model(input_ids=ids, attention_mask=att, labels=lab).loss
    assert torch.allclose(ours, full, atol=1e-5), (ours.item(), full.item())


def test_gradients_flow_and_match():
    model = tiny_model()
    ids, att, lab = batch([10, 6, 10])
    last_token_loss(model, ids, att, lab).backward()
    g1 = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    model(input_ids=ids, attention_mask=att, labels=lab).loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.allclose(g1[n], p.grad, atol=1e-5), n


def test_rejects_an_unmasked_prompt_or_missing_answer():
    model = tiny_model()
    ids, att, lab = batch([8, 8])
    bad = lab.clone()
    bad[0, 2] = 7                                          # a prompt position that is not masked
    with pytest.raises(ValueError):
        last_token_loss(model, ids, att, bad)
    bad = lab.clone()
    bad[1, -1] = -100                                      # no answer token
    with pytest.raises(ValueError):
        last_token_loss(model, ids, att, bad)


def test_registered_training_keywords_are_accepted_by_the_installed_transformers(tmp_path):
    """The G9 recipe's TrainingArguments keywords and the model load keyword must exist in this transformers version
    (transformers 5 removed warmup_ratio; the smoke test of 2026-10-04 caught it on the GPU server)."""
    from types import SimpleNamespace

    from src.confrec.train_lora_yesno import training_arguments
    ns = SimpleNamespace(out=str(tmp_path / "out"), bsz=8, grad_accum=4, epochs=1.0, lr=1e-4, seed=0)
    kw = training_arguments(ns)
    args = transformers.TrainingArguments(**{**kw, "bf16": False, "use_cpu": True})       # bf16 needs a GPU or CPU flags
    assert args.lr_scheduler_type == "cosine" and args.gradient_accumulation_steps == 4
    assert float(args.warmup_steps) == pytest.approx(0.03) and args.learning_rate == 1e-4
    tiny_model().save_pretrained(tmp_path / "tiny")
    model = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "tiny", dtype=torch.bfloat16)
    assert next(model.parameters()).dtype == torch.bfloat16


def test_num_items_in_batch_follows_the_stock_rule():
    model = tiny_model()
    ids, att, lab = batch([12, 7, 9, 12])
    with torch.no_grad():
        mean = last_token_loss(model, ids, att, lab)
        window = last_token_loss(model, ids, att, lab, num_items_in_batch=8)    # 2 micro-batches of 4 labels
    assert torch.allclose(window, mean * 4 / 8, atol=1e-6)


class Rows(torch.utils.data.Dataset):
    """Left-paddable `prompt + answer` rows of different lengths, labels only on the answer token."""

    def __init__(self, lengths, vocab=300, seed=3):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for n in lengths:
            seq = torch.randint(5, vocab, (n,), generator=g).tolist()
            self.items.append({"input_ids": seq, "labels": [-100] * (n - 1) + [seq[-1]]})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def lora_run(trainer_cls, out_dir, grad_accum, **train_kw):
    """Two optimizer steps of LoRA training on the tiny model; returns the LoRA parameters afterwards. Plain SGD without
    clipping by default: the update is then linear in the gradient, so a wrongly scaled loss (e.g. `grad_accum` times too
    large) moves the weights visibly, while Adam's per-parameter normalisation would turn float noise on near-zero
    gradients into sign flips."""
    peft = pytest.importorskip("peft")
    model = tiny_model().train()
    torch.manual_seed(0)                                   # the LoRA A matrices are drawn from the global RNG
    model = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
                                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    kw = dict(output_dir=str(out_dir), per_device_train_batch_size=2, gradient_accumulation_steps=grad_accum,
              max_steps=2, optim="sgd", learning_rate=0.5, max_grad_norm=1e9, lr_scheduler_type="constant",
              weight_decay=0.0, save_strategy="no", report_to=[], seed=0, data_seed=0, remove_unused_columns=False,
              use_cpu=True, disable_tqdm=True, logging_steps=1000)
    kw.update(train_kw)
    trainer_cls(model=model, args=transformers.TrainingArguments(**kw), train_dataset=Rows([9, 5, 8, 6, 7, 9, 4, 8]),
                data_collator=lambda b: collate(b, 0)).train()
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_trainer_updates_equal_the_stock_trainers(tmp_path, grad_accum):
    stock = lora_run(transformers.Trainer, tmp_path / "stock", grad_accum)   # full-sequence loss, labels on the answer
    ours = lora_run(last_token_trainer(transformers.Trainer), tmp_path / "ours", grad_accum)
    assert stock.keys() == ours.keys() and stock
    assert max(float(p.abs().max()) for n, p in stock.items() if "lora_B" in n) > 1e-3     # the run did train
    for n in stock:
        assert torch.allclose(stock[n], ours[n], atol=1e-5, rtol=1e-4), n
