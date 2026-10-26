# Learning notes

This directory is a durable technical learning record, not a dump of every command used during development.

Create a note when a concept is central, recurring, has meaningful trade-offs, caused confusion/bugs, or is relevant to ML/CV interviews.

## Organization

Prefer a lightweight index plus focused topic notes over one very long document.

A learning area should normally have:

```text
area/
├── README.md       # map, reading order and high-level mental model
└── topics/         # focused notes that can be reviewed independently
```

Keep related concepts together when they make sense to study as one unit. Do not create one file per tiny term just for the sake of granularity.

The goal is that, months later, a concept can be recovered directly: if the question is about tensors, open the tensor note; if it is about pooling, open the pooling note.

## Mathematics

Use GitHub-compatible Markdown math consistently:

- inline math: `$...$`;
- display math: fenced `math` blocks;
- do not introduce `\(...\)`, `\[...\]` or raw `$$...$$` blocks in canonical notes.

For example:

````markdown
```math
\frac{1}{N}\sum_{i=1}^{N} x_i
```
````

GitHub supports both `$$...$$` and fenced `math` syntax, but fenced `math` blocks are less ambiguous for multiline equations inside `.md` files and are therefore the repository convention.

Prefer simple MathJax macros that GitHub renders reliably. Avoid macros known to be rejected by GitHub's renderer, such as `\operatorname`; use simpler forms such as `\mathrm{ReLU}` instead.

Important equations should be readable, not only correct. Define non-trivial notation near first use and follow important equations with a plain-language interpretation.

Use LaTeX for actual mathematical relationships such as activation functions, losses, derivatives and equations. Keep tensor shapes such as `[B, C, H, W]` as code when the goal is to describe a software data structure rather than write a mathematical equation.

Run:

```bash
python scripts/check_markdown_math.py
```

to validate the repository math conventions locally. CI runs the same check.

## Visual explanations

When a concept is spatial, graphical, or hard to picture from prose alone, add a visual if it materially improves understanding.

Use the simplest useful medium:

- Mermaid or Markdown diagrams for flows and tensor shapes;
- matplotlib plots for functions, distributions, metrics and experiment results;
- notebook visualizations for real images, masks and overlays;
- small self-contained HTML/CSS/JavaScript demos when interaction genuinely helps, such as a convolution kernel sliding across an image.

Visuals should explain a mechanism, relationship or result. Do not add frontend libraries or dependencies only for decoration.
