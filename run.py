import os
import sys
import torch
from model import GPTLanguageModel, TrainConfig
from evaluate import evaluate_math_correct, evaluate_bool_correct
from utils import (
    get_device,
    build_stoi_itos,
    make_encode_decode,
    make_test_samples,
    show_prompt_result,
)

device = get_device()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


def run_part1_demo():

    print("PART 1: MATH GPT DEMONSTRATION")

    # Setup vocabulary
    math_chars = list("0123456789+-*/()= \n")
    m_stoi, m_itos = build_stoi_itos(math_chars)
    m_enc, m_dec = make_encode_decode(m_stoi, m_itos)
    vocab_size = len(math_chars)

    # Load model
    model_path = "./model_weights_part1.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_part1.py first!")
        return

    # Use medium config (adjust based on your best model)
    cfg = TrainConfig(n_embd=128, n_head=4, n_layer=4, block_size=64)
    model = GPTLanguageModel(cfg, vocab_size, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded from: {model_path}")

    # Test on various datasets
    test_files = [
        ("Addition", "math/test/math_test_addition.txt"),
        ("Simple (+ and -)", "math/test/math_test_simple.txt"),
        ("All Operations", "math/test/math_test_all_ops.txt"),
        ("Complex (with ())", "math/test/math_test_complex.txt"),
    ]

    for name, test_file in test_files:
        test_path = os.path.join(DATA_DIR, test_file)
        if os.path.exists(test_path):
            test_samples = make_test_samples(test_path, num_samples=1000)
            acc = evaluate_math_correct(
                model, m_enc, m_dec, device, test_samples, num_samples=1000
            )
            print(f"{name:20s}: {acc:6.2f}%")

    # Show example predictions
    print("\nExample Predictions:")
    examples = ["1+1=", "10+5=", "9-3=", "6*7=", "8/2=", "(2+3)*4="]
    for ex in examples:
        result = show_prompt_result(model, ex, m_enc, m_dec, device, max_new_tokens=16)
        print(f"  {result}")


def run_part2_demo():
    """Demonstrate Part 2: Boolean Logic GPT"""

    print("PART 2: BOOLEAN LOGIC GPT")

    # Setup vocabulary - build from training file
    train_path = os.path.join(DATA_DIR, "bool", "bool_train.txt")
    test_path = os.path.join(DATA_DIR, "bool", "bool_test_mixed.txt")

    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(test_path, "r", encoding="utf-8") as f:
        test_text = f.read()

    bool_chars = sorted(list(set(train_text + test_text)))
    b_stoi, b_itos = build_stoi_itos(bool_chars)
    b_enc, b_dec = make_encode_decode(b_stoi, b_itos)
    vocab_size = len(bool_chars)

    # Load model
    model_path = "./model_weights_part2.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run train_part2.py first!")
        return

    # Use medium config (adjust based on your best model)
    cfg = TrainConfig(n_embd=128, n_head=4, n_layer=4, block_size=64)
    model = GPTLanguageModel(cfg, vocab_size, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Model loaded from: {model_path}")

    # Test on mixed and by depth
    test_mixed = os.path.join(DATA_DIR, "bool", "bool_test_mixed.txt")
    mixed_samples = make_test_samples(test_mixed, num_samples=1000)
    mixed_acc = evaluate_bool_correct(
        model, b_enc, b_dec, device, mixed_samples, num_samples=1000
    )
    print(f"Mixed Test Accuracy: {mixed_acc:.2f}%")

    print("\nAccuracy by Depth:")
    for d in range(1, 7):
        depth_file = os.path.join(DATA_DIR, "bool", f"bool_test_depth_{d}.txt")
        if os.path.exists(depth_file):
            samples = make_test_samples(depth_file, num_samples=1000)
            acc = evaluate_bool_correct(
                model, b_enc, b_dec, device, samples, num_samples=len(samples)
            )
            print(f"  Depth {d}: {acc:6.2f}%")

    # Show example predictions
    print("\nExample Predictions:")
    examples = ["T=", "F=", "T|F=", "T&F=", "(!T)=", "(T^F)=", "(T|((!F)))="]
    for ex in examples:
        result = show_prompt_result(model, ex, b_enc, b_dec, device, max_new_tokens=16)
        print(f"  {result}")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        part = sys.argv[1]
        if part == "1" or part == "part1":
            run_part1_demo()
        elif part == "2" or part == "part2":
            run_part2_demo()
        else:
            print("Usage: python run.py [1|2|part1|part2]")
            print("  Or run without arguments to demonstrate both parts")
    else:
        # Run both
        run_part1_demo()
        run_part2_demo()


if __name__ == "__main__":
    main()
