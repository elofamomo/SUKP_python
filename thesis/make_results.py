"""
Builds the Chapter 4 benchmark tables from the result CSVs shown in the slides.

Inputs (run from the repo root):
    results_csv/profit.csv       literature results and our profit per benchmark instance (one 30 s run)
Outputs:
    thesis/chapter4/tables/set1.tex, set2.tex, and a summary on stdout.

Usage:
    python thesis/make_results.py
"""
import csv
import os
import numpy as np

OUT_TAB = 'thesis/chapter4/tables'
LIT = ['babc', 'i2pls', 'kbts', 'e2ls', 'ql', 'gbpso', 'aes_bm']
HEAD = {'babc': 'BABC', 'ql': 'QL', 'gbpso': 'gbPSO', 'kbts': 'KBTS', 'e2ls': 'E2LS'}


def short(name):
    """sukp_100_85_0.10_0.75.txt -> ('100', '85', '0.10', '0.75')"""
    return tuple(os.path.splitext(name)[0].split('_')[1:])


def load():
    rows = {}
    with open('results_csv/profit.csv') as f:
        for r in csv.DictReader(f):
            vals = {k: int(r[k]) for k in LIT if r[k]}
            rows[r['instance']] = {'set': r['set'], 'vals': vals, 'bks': max(vals.values()), 'ours': int(r['profit'])}
    return rows


def order(rows, set_name):
    """Instances of a set in the slides' order: by (alpha, beta), then size, then m."""
    names = [k for k, v in rows.items() if v['set'] == set_name]
    return sorted(names, key=lambda k: (short(k)[2], int(short(k)[0]) + int(short(k)[1]), int(short(k)[0])))


def bold_if(v, best):
    return f'\\textbf{{{v}}}' if v == best else f'{v}'


def instance_table(rows, set_name, cols, path, caption, label):
    short_caption = f'Kết quả trên {set_name}'
    lines = [r'\begin{table}[H]', r'\centering', f'\\caption[{short_caption}]{{{caption}}}', f'\\label{{{label}}}', r'\small',
             r'\begin{tabular}{l' + 'r' * (len(cols) + 3) + '}', r'\toprule',
             ' & '.join([r'$m\times n$'] + [HEAD[c] for c in cols] + ['BKS', 'Luận văn', r'$\Delta$']) + r'\\']
    current = None
    for name in order(rows, set_name):
        m, n, a, b = short(name)
        if (a, b) != current:
            current = (a, b)
            lines.append(r'\midrule')
            lines.append(f'\\multicolumn{{{len(cols) + 4}}}{{l}}{{$\\alpha={a},\\ \\beta={b}$}}\\\\')
        r = rows[name]
        cells = [f'{m}$\\times${n}'] + [bold_if(r['vals'][c], r['bks']) if c in r['vals'] else '--' for c in cols]
        delta = r['ours'] - r['bks']
        cells += [f'{r["bks"]}', bold_if(r['ours'], r['bks']), '0' if delta == 0 else f'${delta}$']
        lines.append(' & '.join(cells) + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    os.makedirs(OUT_TAB, exist_ok=True)
    rows = load()
    instance_table(rows, 'Set I', ['babc', 'ql', 'gbpso'], f'{OUT_TAB}/set1.tex',
                   'Kết quả trên Set I. BABC, QL, gbPSO: số liệu công bố. BKS: lời giải tốt nhất trong các công bố được so sánh '
                   '(I2PLS, KBTS, E2LS và AES-BM đều đạt BKS trên toàn bộ Set I). Luận văn: một lần chạy 30 giây với luật tỉ số. '
                   r'$\Delta$ = Luận văn $-$ BKS. In đậm: bằng BKS.', 'tab:set1')
    instance_table(rows, 'Set II', ['kbts', 'e2ls'], f'{OUT_TAB}/set2.tex',
                   'Kết quả trên Set II. KBTS, E2LS: số liệu công bố; BKS: giá trị lớn nhất của hai thuật toán. '
                   r'Luận văn: một lần chạy 30 giây với luật tỉ số. $\Delta$ = Luận văn $-$ BKS. In đậm: bằng BKS.', 'tab:set2')

    for set_name in ('Set I', 'Set II'):
        names = order(rows, set_name)
        gaps = [100.0 * (rows[k]['bks'] - rows[k]['ours']) / rows[k]['bks'] for k in names]
        hit = sum(rows[k]['ours'] == rows[k]['bks'] for k in names)
        print(f'{set_name}: BKS on {hit}/{len(names)}, mean gap {np.mean(gaps):.3f}%, max gap {max(gaps):.3f}%')


if __name__ == '__main__':
    main()
