from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
LEGACY_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
INLINE_CODE = re.compile(r"`[^`]*`")


def unescaped_dollar_count(text: str) -> int:
    count = 0
    for index, char in enumerate(text):
        if char != "$":
            continue
        if index > 0 and text[index - 1] == "\\":
            continue
        count += 1
    return count


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    fence_marker: str | None = None
    in_display_math = False

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        marker = None
        if stripped.startswith("```"):
            marker = "```"
        elif stripped.startswith("~~~"):
            marker = "~~~"

        if marker:
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
            continue

        if in_fence:
            continue

        visible = INLINE_CODE.sub("", line)

        for delimiter in LEGACY_DELIMITERS:
            if delimiter in visible:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: use $...$ or $$...$$ instead of {delimiter}"
                )

        display_count = visible.count("$$")
        if display_count % 2 == 1:
            in_display_math = not in_display_math

        without_display = visible.replace("$$", "")
        if not in_display_math and unescaped_dollar_count(without_display) % 2 == 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: unbalanced inline $ delimiter"
            )

    if in_display_math:
        errors.append(f"{path.relative_to(ROOT)}: unclosed $$ display-math block")

    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(DOCS_DIR.rglob("*.md")):
        errors.extend(check_file(path))

    if errors:
        print("Markdown math check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Markdown math check OK: canonical $ / $$ delimiters are balanced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
