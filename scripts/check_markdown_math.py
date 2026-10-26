from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
LEGACY_DELIMITERS = (r"\(", r"\)", r"\[", r"\]")
FORBIDDEN_GITHUB_MACROS = (r"\operatorname",)
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


def check_forbidden_math(path: Path, line_number: int, text: str) -> list[str]:
    errors: list[str] = []
    for macro in FORBIDDEN_GITHUB_MACROS:
        if macro in text:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: {macro} is not supported by GitHub math rendering; use a simpler macro such as \\mathrm"
            )
    return errors


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    fence_marker: str | None = None
    fence_language: str | None = None

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
                fence_language = stripped[len(marker):].strip()
            elif marker == fence_marker:
                in_fence = False
                fence_marker = None
                fence_language = None
            continue

        if in_fence:
            if fence_language == "math":
                errors.extend(check_forbidden_math(path, line_number, line))
                if "$$" in line:
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line_number}: do not use $$ inside a ```math block"
                    )
            continue

        visible = INLINE_CODE.sub("", line)

        for delimiter in LEGACY_DELIMITERS:
            if delimiter in visible:
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_number}: use $...$ inline math or a ```math block instead of {delimiter}"
                )

        if "$$" in visible:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: use a fenced ```math block for display equations instead of $$"
            )

        errors.extend(check_forbidden_math(path, line_number, visible))

        if unescaped_dollar_count(visible) % 2 == 1:
            errors.append(
                f"{path.relative_to(ROOT)}:{line_number}: unbalanced inline $ delimiter"
            )

    if in_fence and fence_language == "math":
        errors.append(f"{path.relative_to(ROOT)}: unclosed ```math block")

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

    print("Markdown math check OK: GitHub-compatible inline math and fenced math blocks validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
