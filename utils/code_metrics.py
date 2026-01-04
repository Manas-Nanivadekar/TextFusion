import ast
import subprocess
from typing import List, Dict
import tempfile
from pathlib import Path


def check_syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def compute_syntax_validity(samples: List[str]) -> Dict[str, float]:
    valid_count = 0
    syntax_errors = []

    for i, code in enumerate(samples):
        if check_syntax_valid(code):
            valid_count += 1
        else:
            try:
                ast.parse(code)
            except SyntaxError as e:
                syntax_errors.append(
                    {"sample_idx": i, "error": str(e), "line": e.lineno}
                )

    return {
        "valid_count": valid_count,
        "total": len(samples),
        "validity_rate": valid_count / len(samples) if samples else 0.0,
        "syntax_errors": syntax_errors[:5],
    }


def check_execution_safe(code: str, timeout: int = 5) -> Dict[str, any]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        result = subprocess.run(
            ["python", temp_path], capture_output=True, text=True, timeout=timeout
        )

        success = result.returncode == 0

        return {
            "success": success,
            "returncode": result.returncode,
            "stdout": result.stdout[:200],
            "stderr": result.stderr[:200] if result.stderr else "",
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "timeout",
            "message": f"Execution exceeded {timeout}s",
        }
    except Exception as e:
        return {"success": False, "error": "exception", "message": str(e)}
    finally:
        Path(temp_path).unlink(missing_ok=True)


def analyze_code_structure(code: str) -> Dict[str, any]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"valid": False}

    stats = {
        "valid": True,
        "num_functions": 0,
        "num_classes": 0,
        "num_imports": 0,
        "num_assignments": 0,
        "num_loops": 0,
        "num_conditionals": 0,
        "lines": len(code.split("\n")),
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            stats["num_functions"] += 1
        elif isinstance(node, ast.ClassDef):
            stats["num_classes"] += 1
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            stats["num_imports"] += 1
        elif isinstance(node, ast.Assign):
            stats["num_assignments"] += 1
        elif isinstance(node, (ast.For, ast.While)):
            stats["num_loops"] += 1
        elif isinstance(node, ast.If):
            stats["num_conditionals"] += 1

    return stats


if __name__ == "__main__":
    print("Testing code metrics...")

    valid_code = """
def hello(name):
    return f"Hello, {name}!"

class Greeter:
    def __init__(self):
        self.count = 0
    
    def greet(self):
        self.count += 1
        return "Hi!"
"""

    print("\n1. Valid code:")
    print(f"  Syntax valid: {check_syntax_valid(valid_code)}")

    structure = analyze_code_structure(valid_code)
    print(f"  Structure: {structure}")

    invalid_code = """
def broken(
    return "oops"
"""

    print("\n2. Invalid code:")
    print(f"  Syntax valid: {check_syntax_valid(invalid_code)}")

    # Test batch
    samples = [
        valid_code,
        invalid_code,
        "print('hello')",
        "x = 1 + ",  # Invalid
        "import torch\n\nx = torch.tensor([1, 2, 3])",
    ]

    print("\n3. Batch validation:")
    validity = compute_syntax_validity(samples)
    print(f"  Valid: {validity['valid_count']}/{validity['total']}")
    print(f"  Validity rate: {validity['validity_rate']:.2%}")

    if validity["syntax_errors"]:
        print(f"  First error: {validity['syntax_errors'][0]}")

    print("\nCode metrics work!")
