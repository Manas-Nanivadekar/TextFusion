import re
from typing import Dict, List
import random


def create_infilling_example(
    code: str, mask_ratio_range: tuple = (0.2, 0.5)
) -> Dict[str, str]:
    lines = code.split("\n")

    if len(lines) < 3:
        return {"before": "", "masked": code, "after": "", "full": code}

    mask_ratio = random.uniform(*mask_ratio_range)
    num_to_mask = max(1, int(len(lines) * mask_ratio))

    max_start = len(lines) - num_to_mask
    start_idx = random.randint(0, max(0, max_start))
    end_idx = start_idx + num_to_mask

    before_lines = lines[:start_idx]
    masked_lines = lines[start_idx:end_idx]
    after_lines = lines[end_idx:]

    return {
        "before": "\n".join(before_lines),
        "masked": "\n".join(masked_lines),
        "after": "\n".join(after_lines),
        "full": code,
    }


def smart_infilling_example(code: str) -> Dict[str, str]:
    lines = code.split("\n")

    indents = []
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            indents.append(indent)
        else:
            indents.append(0)

    blocks = []
    current_block = [0]
    current_indent = indents[0] if indents else 0

    for i, indent in enumerate(indents[1:], 1):
        if indent == current_indent and lines[i].strip():
            current_block.append(i)
        else:
            if len(current_block) > 0:
                blocks.append(current_block)
            current_block = [i]
            current_indent = indent

    if current_block:
        blocks.append(current_block)

    if not blocks:
        return create_infilling_example(code)

    block_to_mask = random.choice(blocks)
    start_idx = min(block_to_mask)
    end_idx = max(block_to_mask) + 1

    before_lines = lines[:start_idx]
    masked_lines = lines[start_idx:end_idx]
    after_lines = lines[end_idx:]

    return {
        "before": "\n".join(before_lines),
        "masked": "\n".join(masked_lines),
        "after": "\n".join(after_lines),
        "full": code,
    }


if __name__ == "__main__":

    code = """def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total"""

    print("Testing simple infilling...")
    example = create_infilling_example(code, mask_ratio_range=(0.3, 0.4))

    print(f"\nBefore:\n{example['before']}")
    print(f"\n{'='*50}")
    print(f"Masked:\n{example['masked']}")
    print(f"{'='*50}")
    print(f"\nAfter:\n{example['after']}")

    print("\n\nTesting variety (5 examples):")
    for i in range(5):
        ex = create_infilling_example(code)
        print(
            f"\nExample {i+1}: Mask lines {code.split(chr(10)).index(ex['masked'].split(chr(10))[0]) if ex['masked'] else 'N/A'}"
        )

    print("\n\nTesting smart infilling...")
    smart_ex = smart_infilling_example(code)
    print(f"\nBefore:\n{smart_ex['before']}")
    print(f"\n{'='*50}")
    print(f"Masked:\n{smart_ex['masked']}")
    print(f"{'='*50}")
    print(f"\nAfter:\n{smart_ex['after']}")

    print("\n Infilling works!")
