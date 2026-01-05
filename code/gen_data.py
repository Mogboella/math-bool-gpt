import random
import os


def addition(max_num):
    a = random.randint(0, max_num)
    b = random.randint(0, max_num)
    val = a + b

    return f"{a}+{b}={val}"


def subtraction(max_num):
    a = random.randint(0, max_num)
    b = random.randint(0, a)
    val = a - b

    # getting b from 0 to a keeps val
    # positive for dataset simplicity

    return f"{a}-{b}={val}"


def multiplication(max_num):
    a = random.randint(0, max_num)
    b = random.randint(0, max_num)
    val = a * b

    return f"{a}*{b}={val}"


def division(max_num):
    b = random.randint(1, max_num)
    a = b * random.randint(0, max_num)
    val = a // b

    return f"{a}/{b}={val}"


def generate_expression(complexity, max_num):
    if complexity == 0:
        val = random.randint(1, max_num)
        return str(val), val

    ops = {
        "+": lambda x, y: x + y,
        "-": lambda x, y: x - y,
        "*": lambda x, y: x * y,
    }
    operations = random.choice(["+", "-", "*", "/"])

    left_complexity = random.randint(0, complexity - 1)
    right_complexity = complexity - 1 - left_complexity

    if operations == "/":
        right_str, right_val = generate_expression(right_complexity, max_num)
        if right_val == 0:
            right_val = 1
            right_str = "1"

        target_result = random.randint(0, max_num)

        left_val = target_result * right_val
        left_str = f"({left_val})"

        expr_str = f"({left_str}/{right_str})"
        return expr_str, target_result

    else:
        left_str, left_val = generate_expression(left_complexity, max_num)
        right_str, right_val = generate_expression(right_complexity, max_num)

        expr_str = f"({left_str}{operations}{right_str})"
        val = ops[operations](left_val, right_val)

        return expr_str, val


def parentheses_expression(max_num, complexity=2):
    expr_str, val = generate_expression(complexity, max_num)
    return f"{expr_str}={val}"


def generate_bool_expression(depth):
    if depth == 0 or (depth > 0 and random.random() < 0.2):
        return random.choice(["T", "F"])

    choice = random.choice(["unary", "binary"])

    if choice == "unary":
        sub_expr = generate_bool_expression(depth - 1)
        return f"(!({sub_expr}))"
    else:
        ops = ["&", "|", "^"]
        operation = random.choice(ops)

        left_depth = random.randint(0, depth - 1)
        right_depth = depth - 1 - left_depth

        left_expr = generate_bool_expression(left_depth)
        right_expr = generate_bool_expression(right_depth)

        return f"({left_expr}{operation}{right_expr})"


def evaluate_expression(expr):
    expr = expr.replace("!", " not ")
    expr = expr.replace("&", " and ")
    expr = expr.replace("|", " or ")
    expr = expr.replace("^", " != ")
    expr = expr.replace("T", " True ")
    expr = expr.replace("F", " False ")

    return eval(expr)


def generate_math_dataset(
    num_samples,
    max_num,
    filename,
    operations=[
        addition,
        subtraction,
        multiplication,
        division,
        parentheses_expression,
    ],
):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        for _ in range(num_samples):
            operation = random.choice(operations)
            problem = operation(max_num)
            f.write(problem + "\n")


def generate_bool_dataset(num_samples, max_depth, filename):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        samples = set()
        while len(samples) < num_samples:
            current_depth = random.randint(1, max_depth)
            expr = generate_bool_expression(current_depth)

            if expr not in samples:
                val = evaluate_expression(expr)
                result = "T" if val else "F"
                samples.add(f"{expr}={result}")

        f.write("\n".join(samples))


if __name__ == "__main__":

    # Stage 1: Single-digit addition"
    generate_math_dataset(
        num_samples=10000,
        max_num=9,
        filename="./data/math/train/math_train_addition.txt",
        operations=[addition],
    )
    generate_math_dataset(
        num_samples=2000,
        max_num=9,
        filename="./data/math/test/math_test_addition.txt",
        operations=[addition],
    )

    # Stage 2: Addition + Subtraction"
    generate_math_dataset(
        num_samples=10000,
        max_num=9,
        filename="./data/math/train/math_train_simple.txt",
        operations=[addition, subtraction],
    )
    generate_math_dataset(
        num_samples=2000,
        max_num=9,
        filename="./data/math/test/math_test_simple.txt",
        operations=[addition, subtraction],
    )

    # Stage 3: All basic operations (no parentheses)
    generate_math_dataset(
        num_samples=15000,
        max_num=9,
        filename="./data/math/train/math_train_all_ops.txt",
        operations=[addition, subtraction, multiplication, division],
    )
    generate_math_dataset(
        num_samples=2000,
        max_num=9,
        filename="./data/math/test/math_test_all_ops.txt",
        operations=[addition, subtraction, multiplication, division],
    )

    # Stage 4: Expressions with parentheses
    generate_math_dataset(
        num_samples=15000,
        max_num=9,
        filename="./data/math/train/math_train_complex.txt",
        operations=[
            addition,
            subtraction,
            multiplication,
            division,
            parentheses_expression,
        ],
    )
    generate_math_dataset(
        num_samples=2000,
        max_num=9,
        filename="./data/math/test/math_test_complex.txt",
        operations=[
            addition,
            subtraction,
            multiplication,
            division,
            parentheses_expression,
        ],
    )

    # Training set: mixed depths
    generate_bool_dataset(10000, 6, "./data/bool/bool_train.txt")

    # Test set: mixed depths
    generate_bool_dataset(1000, 6, "./data/bool/bool_test_mixed.txt")

    # Test sets by specific depth (for analysis)
    for d in range(1, 7):
        num_samples = 200 if d > 2 else 10  # Fewer samples for very deep expressions
        generate_bool_dataset(num_samples, d, f"./data/bool/bool_test_depth_{d}.txt")
