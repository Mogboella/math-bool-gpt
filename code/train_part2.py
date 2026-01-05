import os
import time
import torch
from model import GPTLanguageModel, TrainConfig
from utils import (
    set_seed,
    get_device,
    build_stoi_itos,
    make_encode_decode,
    make_train_val_data,
    make_get_batch,
    estimate_loss,
    make_test_samples,
    sample_generations,
    show_prompt_result,
)
from evaluate import evaluate_bool_correct

set_seed(1337)
device = get_device()
print(f"Using device: {device}")

BOOL_CHARS_TEXT = []
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "bool")
TEST_DIR = os.path.join(BASE_DIR, "data", "bool")

# Build vocab from actual training file
with open(os.path.join(TRAIN_DIR, "bool_train.txt"), "r", encoding="utf-8") as f:
    train_text = f.read()
with open(os.path.join(TEST_DIR, "bool_test_mixed.txt"), "r", encoding="utf-8") as f:
    test_text = f.read()

BOOL_CHARS = sorted(list(set(train_text + test_text)))
stoi, itos = build_stoi_itos(BOOL_CHARS)
encode, decode = make_encode_decode(stoi, itos)
vocab_size = len(BOOL_CHARS)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {BOOL_CHARS}")

# EMBEDDING SIZE EXPERIMENTS

CFG_EMBED_32 = TrainConfig(
    n_embd=32,  # Very small
    n_head=2,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
)

CFG_EMBED_64 = TrainConfig(
    n_embd=64,  # Baseline
    n_head=2,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
)

CFG_EMBED_128 = TrainConfig(
    n_embd=128,
    n_head=4,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
)

# NUMBER OF HEADS EXPERIMENTS

CFG_HEAD_1 = TrainConfig(
    n_embd=64,
    n_head=1,  # Single head
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
)

CFG_HEAD_2 = TrainConfig(
    n_embd=64,
    n_head=2,  # Baseline
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
)

# NUMBER OF LAYERS EXPERIMENTS


CFG_LAYER_2 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,  # Baseline
    block_size=64,
    batch_size=64,
    max_iters=1500,
)

CFG_LAYER_4 = TrainConfig(
    n_embd=64, n_head=2, n_layer=4, block_size=64, batch_size=64, max_iters=1500  # Deep
)

# BLOCK SIZE (CONTEXT LENGTH) EXPERIMENTS

CFG_BLOCK_32 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=32,  # Short context
    batch_size=64,
    max_iters=1500,
)

CFG_BLOCK_64 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=64,  # Baseline
    batch_size=64,
    max_iters=1500,
)

CFG_BLOCK_256 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=256,  # Very long context
    batch_size=64,
    max_iters=1500,
)

# DROPOUT EXPERIMENTS

CFG_DROPOUT_0 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
    dropout=0.0,  # No regularization
)

CFG_DROPOUT_01 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
    dropout=0.1,  # Light regularization
)

CFG_DROPOUT_02 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
    dropout=0.2,  # Standard
)

CFG_DROPOUT_03 = TrainConfig(
    n_embd=64,
    n_head=2,
    n_layer=2,
    block_size=64,
    batch_size=64,
    max_iters=1500,
    dropout=0.3,  # Heavy regularization
)

# CFGs

CFG_SMALL = TrainConfig(
    n_embd=64, n_head=2, n_layer=2, block_size=64, batch_size=64, max_iters=1500
)
CFG_MEDIUM = TrainConfig(
    n_embd=128, n_head=4, n_layer=4, block_size=64, batch_size=64, max_iters=2000
)
CFG_LONGCTX = TrainConfig(
    n_embd=128, n_head=4, n_layer=4, block_size=128, batch_size=64, max_iters=2000
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(
    name: str,
    train_path: str,
    test_path: str,
    cfg: TrainConfig,
    test_num_samples: int = 200,
    print_samples: bool = True,
):
    """
    Train a Boolean Logic GPT model.

    Args:
        name: Experiment name
        train_path: Path to training data
        test_path: Path to test data
        cfg: Training configuration
        test_num_samples: Number of samples for intermediate testing
        print_samples: Whether to print sample outputs

    Returns:
        Dictionary with training results
    """
    print(f"Training: {name}")

    # Load data
    train_data, val_data = make_train_val_data(encode, train_path, train_fraction=0.8)
    get_batch = make_get_batch(train_data, val_data, cfg, device)
    test_samples = make_test_samples(test_path, num_samples=2000)

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    print(f"Test samples: {len(test_samples)}")

    # Initialize model
    model = GPTLanguageModel(cfg, vocab_size, device)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    # Training loop
    t0 = time.time()
    for it in range(cfg.max_iters):
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            losses = estimate_loss(model, get_batch, cfg.eval_iters, device)
            acc = evaluate_bool_correct(
                model,
                encode,
                decode,
                device,
                test_samples,
                num_samples=test_num_samples,
                max_new_tokens=cfg.max_new_tokens,
            )
            print(
                f"Iter {it:4d}/{cfg.max_iters} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f} | "
                f"test_acc@{test_num_samples} {acc:.1f}%"
            )

        # Training step
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # Final evaluation
    losses = estimate_loss(model, get_batch, cfg.eval_iters, device)
    acc = evaluate_bool_correct(
        model,
        encode,
        decode,
        device,
        test_samples,
        num_samples=1000,
        max_new_tokens=cfg.max_new_tokens,
    )

    print(f"\nFinal Results:")
    print(f"  Train Loss: {losses['train']:.4f}")
    print(f"  Val Loss: {losses['val']:.4f}")
    print(f"  Test Accuracy: {acc:.2f}%")

    # Save model
    weight_path = os.path.join(MODEL_DIR, f"{name}.pth")
    torch.save(model.state_dict(), weight_path)
    print(f"  Model saved to: {weight_path}")

    if print_samples:
        print(f"\nSample Generations:")
        prompts = ["T=", "F=", "T|F=", "(T&F)=", "(!T)="]
        samples_out = sample_generations(
            model, prompts, encode, decode, device, max_new_tokens=cfg.max_new_tokens
        )
        for prompt, output in zip(prompts, samples_out):
            first_line = output.split("\n")[0]
            print(f"  {first_line}")
        print("\n")

    return {
        "name": name,
        "train_loss": losses["train"],
        "val_loss": losses["val"],
        "test_acc@1000": acc,
        "seconds": elapsed,
        "weights": weight_path,
        "params": num_params,
    }


def evaluate_by_depth(model, cfg):
    """
    Evaluate model accuracy by expression depth.
    """
    print("EVALUATION BY DEPTH")

    depth_results = {}
    for d in range(1, 7):
        test_file = os.path.join(TEST_DIR, f"bool_test_depth_{d}.txt")
        samples = make_test_samples(test_file, num_samples=1000)
        acc = evaluate_bool_correct(
            model,
            encode,
            decode,
            device,
            samples,
            num_samples=len(samples),
            max_new_tokens=cfg.max_new_tokens,
        )
        depth_results[d] = acc
        print(f"Depth {d}: {acc:.2f}%")
    return depth_results


def run_architectural_experiments_bool():
    """
    Task 2.3: Architectural exploration for Boolean Logic
    """

    train_path = os.path.join(TRAIN_DIR, "bool_train.txt")
    test_path = os.path.join(TEST_DIR, "bool_test_mixed.txt")

    all_results = []

    print("ARCHITECTURAL EXPLORATION - PART 2: BOOL GPT")

    print("\n### EXPERIMENT 1: EMBEDDING SIZE ###")
    embed_experiments = [
        ("embed_32", CFG_EMBED_32),
        ("embed_64", CFG_EMBED_64),
        ("embed_128", CFG_EMBED_128),
    ]

    embed_results = []
    for name, cfg in embed_experiments:
        result = train_model(name, train_path, test_path, cfg, test_num_samples=200)
        embed_results.append(result)
        all_results.append(result)

    print("\n--- Embedding Size Comparison ---")
    print(
        f"{'Size':<10} {'Params':<10} {'Train Loss':<12} {'Val Loss':<12} {'Accuracy':<10} {'Time(s)'}"
    )
    for r in embed_results:
        print(
            f"{r['name']:<10} {r['params']/1e6:<10.2f} "
            f"{r['train_loss']:<12.4f} {r['val_loss']:<12.4f} "
            f"{r['test_acc@1000']:<10.2f} {r['seconds']:<.1f}"
        )

    print("\n### EXPERIMENT 2: NUMBER OF LAYERS ###")
    layer_experiments = [
        ("layer_2", CFG_LAYER_2),
        ("layer_4", CFG_LAYER_4),
    ]

    layer_results = []
    for name, cfg in layer_experiments:
        result = train_model(name, train_path, test_path, cfg, test_num_samples=200)
        layer_results.append(result)
        all_results.append(result)

    print("\n--- Layer Depth Comparison ---")
    print(
        f"{'Layers':<10} {'Params':<10} {'Train Loss':<12} {'Val Loss':<12} {'Accuracy':<10} {'Time(s)'}"
    )
    for r in layer_results:
        print(
            f"{r['name']:<10} {r['params']/1e6:<10.2f} "
            f"{r['train_loss']:<12.4f} {r['val_loss']:<12.4f} "
            f"{r['test_acc@1000']:<10.2f} {r['seconds']:<.1f}"
        )

    print("\n### EXPERIMENT 3: BLOCK SIZE (CONTEXT LENGTH) ###")
    block_experiments = [
        ("block_32", CFG_BLOCK_32),
        ("block_64", CFG_BLOCK_64),
    ]

    block_results = []
    for name, cfg in block_experiments:
        result = train_model(name, train_path, test_path, cfg, test_num_samples=200)
        block_results.append(result)
        all_results.append(result)

    print("\n--- Block Size Comparison ---")
    print(
        f"{'BlockSize':<12} {'Params':<10} {'Train Loss':<12} {'Val Loss':<12} {'Accuracy':<10} {'Time(s)'}"
    )
    for r in block_results:
        print(
            f"{r['name']:<12} {r['params']/1e6:<10.2f} "
            f"{r['train_loss']:<12.4f} {r['val_loss']:<12.4f} "
            f"{r['test_acc@1000']:<10.2f} {r['seconds']:<.1f}"
        )

    print("\n### EXPERIMENT 4: DROPOUT (REGULARIZATION) ###")
    dropout_experiments = [
        ("dropout_0.0", CFG_DROPOUT_0),
        ("dropout_0.1", CFG_DROPOUT_01),
        ("dropout_0.2", CFG_DROPOUT_02),
        ("dropout_0.3", CFG_DROPOUT_03),
    ]

    dropout_results = []
    for name, cfg in dropout_experiments:
        result = train_model(name, train_path, test_path, cfg, test_num_samples=200)
        dropout_results.append(result)
        all_results.append(result)

    print("\n--- Dropout Comparison ---")
    print(
        f"{'Dropout':<10} {'Train Loss':<12} {'Val Loss':<12} {'Overfitting':<12} {'Accuracy':<10}"
    )
    for r in dropout_results:
        overfit = r["val_loss"] - r["train_loss"]
        print(
            f"{r['name']:<10.1f} {r['train_loss']:<12.4f} "
            f"{r['val_loss']:<12.4f} {overfit:<12.4f} {r['test_acc@1000']:<10.2f}"
        )

    return all_results


def run_operation_specific_experiments():
    """
    Task 1.4: Explore different arithmetic operations
    """

    print("OPERATION-SPECIFIC EXPERIMENTS - TASK 2.4")

    # Define experiments
    experiments = [
        (
            "bool_baseline_small",
            os.path.join(TRAIN_DIR, "bool_train.txt"),
            os.path.join(TEST_DIR, "bool_test_mixed.txt"),
            CFG_SMALL,
        ),
        (
            "bool_baseline_medium",
            os.path.join(TRAIN_DIR, "bool_train.txt"),
            os.path.join(TEST_DIR, "bool_test_mixed.txt"),
            CFG_MEDIUM,
        ),
        (
            "bool_longctx",
            os.path.join(TRAIN_DIR, "bool_train.txt"),
            os.path.join(TEST_DIR, "bool_test_mixed.txt"),
            CFG_LONGCTX,
        ),
    ]

    # Run experiments
    results = []
    for name, train_path, test_path, cfg in experiments:
        result = train_model(name, train_path, test_path, cfg)
        results.append(result)

    # Print summary

    print("OPERATION-SPECIFIC TRAINING SUMMARY")

    print(
        f"{'Operation':<25} {'Train Loss':<12} {'Val Loss':<12} {'Test Acc':<12} {'Time (s)'}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['name']:<25} {r['train_loss']:<12.4f} {r['val_loss']:<12.4f} "
            f"{r['test_acc@1000']:<12.2f} {r['seconds']:.1f}"
        )

    return results


if __name__ == "__main__":

    print("PART 2: BOOLEAN LOGIC GPT TRAINING")

    # Task 2.3: Architectural exploration
    # arch_results = run_architectural_experiments_bool()

    # Task 2.4: Operation-specific experiments
    op_results = run_operation_specific_experiments()

    # Save the best overall model
    all_results = op_results
    best_result = max(all_results, key=lambda x: x["test_acc@1000"])
    best_model_path = best_result["weights"]
    final_model_path = os.path.join(BASE_DIR, "model_weights_part2.pth")

    # Copy best model to final location
    import shutil

    shutil.copy(best_model_path, final_model_path)
    print(f"\nBest model ({best_result['name']}) saved as: {final_model_path}")

    # Load best model for detailed evaluation
    best_cfg = CFG_MEDIUM
    best_model = GPTLanguageModel(best_cfg, vocab_size, device).to(device)
    best_model.load_state_dict(torch.load(final_model_path))

    # Evaluate by depth
    depth_results = evaluate_by_depth(best_model, best_cfg)

    # Generate representative examples for appendix
    print("REPRESENTATIVE EXAMPLES FOR APPENDIX")

    test_prompts = [
        "T=",  # simplest sanity
        "F=",  # simplest sanity
        "T|F=",  # OR
        "T&F=",  # AND
        "(!T)=",  # NOT
        "(T^F)=",  # XOR
        "(T|((!F)))=",  # nested
        "((T&F)|T)=",  # complex
    ]

    for prompt in test_prompts:
        result = show_prompt_result(
            best_model, prompt, encode, decode, device, max_new_tokens=16
        )
        print(f"{result}")
