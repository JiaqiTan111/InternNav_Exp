import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def split_lines(md_text: str):
    lines = []
    for raw in md_text.splitlines():
        if raw.strip().startswith('```'):
            continue
        if raw.startswith('# '):
            lines.append(('h1', raw[2:].strip()))
        elif raw.startswith('## '):
            lines.append(('h2', raw[3:].strip()))
        elif raw.startswith('### '):
            lines.append(('h3', raw[4:].strip()))
        elif raw.startswith('- '):
            lines.append(('li', '• ' + raw[2:].strip()))
        elif raw.startswith('1. ') or raw[:3].isdigit() and raw[1:3] == '. ':
            lines.append(('li', raw.strip()))
        elif raw.strip() == '':
            lines.append(('blank', ''))
        else:
            lines.append(('p', raw.strip()))
    return lines


def style_for(kind):
    if kind == 'h1':
        return 17, 'bold'
    if kind == 'h2':
        return 14, 'bold'
    if kind == 'h3':
        return 12, 'bold'
    if kind == 'li':
        return 10.5, 'normal'
    return 10.5, 'normal'


def render(md_path: Path, pdf_path: Path):
    text = md_path.read_text(encoding='utf-8')
    rows = split_lines(text)

    fig_w, fig_h = 8.27, 11.69  # A4 portrait inches
    top_y = 0.96
    left_x = 0.08
    line_gap = 0.028

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        y = top_y

        for kind, content in rows:
            if kind == 'blank':
                y -= line_gap * 0.55
            else:
                size, weight = style_for(kind)
                wrapped = []
                max_chars = 95 if kind in ('p', 'li') else 70
                while len(content) > max_chars:
                    cut = content.rfind(' ', 0, max_chars)
                    if cut <= 0:
                        cut = max_chars
                    wrapped.append(content[:cut].strip())
                    content = content[cut:].strip()
                wrapped.append(content)

                for line in wrapped:
                    if y < 0.06:
                        pdf.savefig(fig)
                        plt.close(fig)
                        fig = plt.figure(figsize=(fig_w, fig_h))
                        ax = fig.add_axes([0, 0, 1, 1])
                        ax.axis('off')
                        y = top_y
                    ax.text(left_x, y, line, fontsize=size, fontweight=weight, va='top', ha='left', family='DejaVu Sans')
                    y -= line_gap

            if kind in ('h1', 'h2', 'h3'):
                y -= line_gap * 0.2

        pdf.savefig(fig)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    render(in_path, out_path)
    print(f'Saved PDF: {out_path}')


if __name__ == '__main__':
    main()
