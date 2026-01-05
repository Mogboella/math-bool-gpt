import torch
from typing import List
from utils import make_test_samples


@torch.no_grad()
def evaluate_math_correct(
    model,
    encode: callable,
    decode: callable,
    device,
    samples: List[str],
    num_samples=100,
    max_new_tokens=20,
):
    model.eval()
    correct = 0

    samples = [s.strip() for s in samples if s.strip()]
    samples = samples[:num_samples]
    n = len(samples)
    if n == 0:
        model.train()
        return 0.0

    for sample in samples:
        if "=" not in sample:
            continue

        lhs, rhs = sample.split("=", 1)
        prompt = lhs + "="

        x = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        y_pred = model.generate(x, max_new_tokens=max_new_tokens)[0]
        full = decode(y_pred.tolist())

        pred = full[len(prompt) :].split("\n")[0].strip()
        if pred == rhs.strip():
            correct += 1

    model.train()
    return 100.0 * correct / n


def evaluate_bool_by_depth(model, paths, cfg):
    depth_results = {}
    for d in range(1, 7):
        test_file = paths[f"test_depth_{d}"]
        samples = make_test_samples(test_file, num_samples=1000)
        acc = evaluate_bool_correct(
            model,
            samples,
            num_samples=len(samples),
            max_new_tokens=cfg.max_new_tokens,
        )
        depth_results[d] = acc
    return depth_results


@torch.no_grad()
def evaluate_bool_correct(
    model,
    encode: callable,
    decode: callable,
    device,
    samples: List[str],
    num_samples=100,
    max_new_tokens=20,
):
    model.eval()
    correct = 0

    samples = [s.strip() for s in samples if s.strip()]
    samples = samples[:num_samples]
    n = len(samples)
    if n == 0:
        model.train()
        return 0.0

    for sample in samples:
        if "=" not in sample:
            continue

        lhs, rhs = sample.split("=", 1)
        prompt = lhs + "="

        x = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        y_pred = model.generate(x, max_new_tokens=max_new_tokens)[0]
        full = decode(y_pred.tolist())

        pred = full[len(prompt) :].split("\n")[0].strip()

        if pred == rhs.strip():
            correct += 1

    model.train()
    return 100.0 * correct / n
