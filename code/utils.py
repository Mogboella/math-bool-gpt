import torch
import os
import random
from typing import List, Iterable, Callable, Dict, Tuple


def set_seed(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )


def build_stoi_itos(chars: Iterable[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create character ↔ integer mappings.
    """
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def make_encode_decode(
    stoi: Dict[str, int],
    itos: Dict[int, str],
) -> Tuple[Callable[[str], List[int]], Callable[[Iterable[int]], str]]:
    """
    Return encode and decode functions.

    encode: string -> list[int]
    decode: list[int] -> string
    """

    def encode(s: str) -> List[int]:
        return [stoi[c] for c in s]

    def decode(indices: Iterable[int]) -> str:
        return "".join(itos[i] for i in indices)

    return encode, decode


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return txt


def as_float(x):
    if torch.is_tensor(x):
        return x.item()
    return float(x)


def pretty_results(res_list):
    rows = []
    for r in res_list:
        rows.append(
            {
                "name": r["name"],
                "train_loss": round(as_float(r["train_loss"]), 4),
                "val_loss": round(as_float(r["val_loss"]), 4),
                "test_acc@1000": round(as_float(r["test_acc@1000"]), 2),
                "seconds": round(as_float(r["seconds"]), 1),
                "weights": r["weights"],
            }
        )
    return rows


def make_train_val_data(encode: callable, train_path: str, train_fraction: float = 0.8):
    text = load_text_file(train_path)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_fraction * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data


def make_test_samples(test_path: str, num_samples: int = 2000) -> List[str]:
    # Evaluator expects lines like "a+b=c"
    with open(test_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[:num_samples]


# data loading
def make_get_batch(
    train_data: torch.Tensor, val_data: torch.Tensor, cfg, device
):
    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([data[i : i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters: int, device):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


@torch.no_grad()
def sample_generations(model, prompts: List[str], encode: callable, decode: callable, device, max_new_tokens=16):
    """
    Generate outputs for a list of prompts.
    Returns list of full generated strings.
    """
    model.eval()
    outputs = []
    for p in prompts:
        x = torch.tensor(encode(p), dtype=torch.long, device=device).unsqueeze(0)
        y = model.generate(x, max_new_tokens=max_new_tokens)[0]
        full = decode(y.tolist())
        outputs.append(full)
    model.train()
    return outputs


def show_prompt_result(model, prompt: str, encode: callable, decode: callable, device, max_new_tokens=16):
    """
    Show the first line of generation for a single prompt.
    """
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
    y = model.generate(x, max_new_tokens=max_new_tokens)[0]
    full = decode(y.tolist())
    first_line = full.split("\n")[0]
    return first_line
