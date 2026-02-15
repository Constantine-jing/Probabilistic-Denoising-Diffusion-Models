# DDPM Lecture Slides — Setup & Explanation Guide

## Quick Start

```bash
# 1. Make sure you have Quarto installed (https://quarto.org/docs/get-started/)
# 2. Make sure you have Python + Jupyter kernel
pip install jupyter numpy matplotlib Pillow

# 3. Preview your slides (opens in browser)
cd quarto-deck
quarto preview index.qmd

# 4. Or render to HTML file
quarto render index.qmd
# Output goes to docs/index.html
```

## File Structure

```
quarto-deck/
├── _quarto.yml          # Project config (tells Quarto this is a project)
├── index.qmd            # ★ YOUR MAIN SLIDES — all content is here
├── references.bib       # Bibliography (auto-generates reference slide)
├── assets/
│   ├── theme.scss        # Custom styling (dark theme, nice fonts)
│   └── demo/
│       └── input.jpg     # (OPTIONAL) Put your own image here for demos
└── README.md             # This file
```

## What Each File Does

### `index.qmd` — The Slide Deck
This is a Quarto Markdown file. It combines:
- **YAML header** (the `---` block at top): Controls slide format, theme, etc.
- **Markdown content**: Your slide text, equations, tables
- **Python code blocks**: Live demos that generate plots
- **Speaker notes**: Inside `::: notes` blocks — only YOU see these during presentation

### Key Quarto Syntax You'll See:

| Syntax | What it does |
|--------|-------------|
| `---` | New slide |
| `## Title` | Slide with a title |
| `# Section Title {background-color="..."}` | Section divider slide |
| `. . .` | Pause (content appears on click) |
| `:::: columns` / `::: column` | Two-column layout |
| `::: notes` | Speaker notes (press S during presentation) |
| `$...$` | Inline math (LaTeX) |
| `$$...$$` | Display math |
| `::: {.key-eq}` | Custom styled box (defined in theme.scss) |
| `{python}` code blocks | Live Python that runs and shows output |
| `#| code-fold: true` | Hides code by default, click to expand |
| `[@ho2020ddpm]` | Citation (auto-linked to references.bib) |

### `references.bib` — Bibliography
BibTeX entries for all papers cited. Quarto auto-generates a References slide.

### `assets/theme.scss` — Styling
Custom dark theme. The `/*-- scss:defaults --*/` section sets variables.
The `/*-- scss:rules --*/` section adds custom CSS classes.

## How to Use Your Own Image

Replace the checkerboard in the demo with a real photo:

1. Put any image at `assets/demo/input.jpg`
2. In `index.qmd`, find the forward diffusion demo code block
3. Replace these lines:
   ```python
   # Replace the checkerboard block with:
   from PIL import Image
   img = Image.open("assets/demo/input.jpg").convert("RGB").resize((128, 128))
   x0 = np.asarray(img).astype(np.float32) / 255.0
   ```

## Presenting

- **Preview**: `quarto preview index.qmd` — live reload as you edit
- **Present**: Open the rendered HTML, press **F** for fullscreen
- **Speaker notes**: Press **S** to open speaker view (shows notes + timer)
- **Navigate**: Arrow keys or click

## What to Customize

1. **Replace "YOUR NAME"** in the YAML header
2. **Add your own image** for the forward process demo
3. **Add/remove slides** as needed for your time budget
4. **The speaker notes** contain full talk tracks — study these!

## Slide Count & Timing

Current deck has ~20 slides. At roughly 1-2 min/slide:
- Methodology section: ~12 slides (~15-20 min) — your teammate might cover some of this
- Applications + Consistency Models: ~6 slides (~10-12 min) — your main part
- Demos + Limitations: ~4 slides (~5-8 min)

Adjust based on how you split with your teammate.
