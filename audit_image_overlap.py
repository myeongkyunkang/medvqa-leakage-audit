import argparse
import csv
import json
import os

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

VQA_RAD_SPLITS = {'train': ('freeform', 'para'), 'test': ('test_freeform', 'test_para')}  # phrase_type values


def load_data(dataset, data_dir, splits):
    """{split: [{'img_name', 'q_lang', 'modality'}]}"""
    if dataset == 'slake':
        return {s: json.load(open(f'{data_dir}/{s}.json', encoding='utf-8')) for s in splits}
    rows = json.load(open(f'{data_dir}/VQA_RAD Dataset Public.json', encoding='utf-8'))
    return {s: [{'img_name': r['image_name'], 'q_lang': 'en', 'modality': r['image_organ']} for r in rows if r['phrase_type'] in VQA_RAD_SPLITS[s]] for s in splits}


def img_id(name):
    return os.path.dirname(name) or os.path.splitext(name)[0]


def feature(path, size):
    """Grayscale image cropped to its nonzero bbox, resized to size x size, zero-mean and unit-norm."""
    img = Image.open(path).convert('L')
    ys, xs = np.nonzero(np.asarray(img))
    if len(ys):  # drop zero padding, specifically for brain MRI
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    vec = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32).ravel()
    vec = vec - vec.mean()
    return vec / max(np.linalg.norm(vec), 1e-6)


def similarity(names, img_dir, size):
    """(N, N) NCC. Diagonal = inf so an identical img_name always matches."""
    feats = np.stack([feature(f'{img_dir}/{n}', size) for n in tqdm(names, desc='Images')])
    sim = feats @ feats.T
    np.fill_diagonal(sim, np.inf)
    return sim


def select_images(data, names, splits, lang_spec, mods):
    """Per-split langs and sorted image indices (into names) for a lang set and modality set."""
    name_to_idx = {n: i for i, n in enumerate(names)}
    # 'en,zh' -> all splits; 'train=en,zh/validation=en/test=en' -> per split
    langs = dict(p.split('=') for p in lang_spec.split('/')) if '=' in lang_spec else dict.fromkeys(splits, lang_spec)
    imgs = {}
    for s in splits:
        selected = set()
        for q in data[s]:
            if q['q_lang'] not in langs[s].split(','):
                continue
            if mods != 'all' and q['modality'] not in mods.split(','):
                continue
            selected.add(name_to_idx[q['img_name']])
        imgs[s] = sorted(selected)
    return langs, imgs


def compute_overlap(names, sim, thr, lang_subsets, splits, show_lang, tag):
    """Returns panels [(labels, count, %)] and per-image best-match rows."""
    panels, rows = [], []
    for lang_spec, langs, imgs in lang_subsets:
        counts = np.zeros((len(splits), len(splits)), int)
        for src_i, src_split in enumerate(splits):
            for dst_i, dst_split in enumerate(splits):
                sim_block = sim[np.ix_(imgs[src_split], imgs[dst_split])]  # (|src|, |dst|)
                hit = sim_block.max(1) >= thr
                counts[src_i, dst_i] = len(imgs[src_split]) if src_i == dst_i else hit.sum()
                for i in (range(len(imgs[src_split])) if src_i != dst_i else []):  # best match per image, any thr
                    match = imgs[dst_split][sim_block[i].argmax()]
                    rows.append([
                        lang_spec,
                        src_split,
                        names[imgs[src_split][i]],
                        dst_split,
                        names[match],
                        round(min(float(sim_block[i].max()), 1.0), 4),
                        int(hit[i])
                    ])
        pct = counts / np.maximum(np.diag(counts)[:, None], 1) * 100
        labels = [f'{s} ({langs[s].upper().replace(",", "/")})' if show_lang else s for s in splits]  # e.g. train (EN/ZH)

        print(f'\n[{tag} | {lang_spec}]')
        for i, label in enumerate(labels):
            print(f'{label:<16}' + ''.join(f'{counts[i, j]:>6d} ({pct[i, j]:5.1f}%)' for j in range(len(splits))))
        panels.append((labels, counts, pct))
    return panels, rows


def plot_overlap(panels, path):
    """Transposed heatmap per panel; only diagonal + leakage (upper right) shown."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 5.2), squeeze=False)
    for ax, (labels, counts, pct) in zip(axes[0], panels):
        counts, pct = counts.T, pct.T
        hide = np.tril(np.ones_like(counts, bool), -1)  # lower left = train -> val/test (reverse direction, not the leakage metric)
        ax.imshow(np.ma.masked_where(hide, pct), cmap='Reds', vmin=0, vmax=100)
        for (i, j), count in np.ndenumerate(counts):
            if not hide[i, j]:  # column j = reference split, denominator = its image count (diagonal)
                ax.text(j, i, f'{count}/{counts[j, j]}\n({pct[i, j]:.1f}%)', ha='center', va='center', fontsize=15, color='white' if pct[i, j] > 60 else 'black')
        ticks = [label.replace(' (', '\n(') for label in labels]  # train (EN/ZH) -> train\n(EN/ZH)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=14)
        ax.set_yticks(range(len(ticks)), ticks, fontsize=14)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
    plt.close(fig)


def save_pairs(pairs, img_dir, vis_dir):
    if not pairs:
        return
    os.makedirs(vis_dir, exist_ok=True)
    title_h = 28
    font = ImageFont.load_default(size=18)
    for src_split, src_name, dst_split, dst_name, ncc in pairs:
        canvas = Image.new('L', (512, 256 + title_h))
        draw = ImageDraw.Draw(canvas)
        for x, split, name in ((0, src_split, src_name), (256, dst_split, dst_name)):
            img = Image.open(f'{img_dir}/{name}').convert('L')
            img.thumbnail((256, 256))
            draw.text((x + 128, 4), f'{img_id(name)} ({split})', fill=255, font=font, anchor='mt')  # e.g. xmlab102 (test)
            canvas.paste(img, (x + (256 - img.width) // 2, title_h + (256 - img.height) // 2))
        canvas.save(f'{vis_dir}/{src_split}_{img_id(src_name)}__{dst_split}_{img_id(dst_name)}_{ncc:.3f}.jpg', quality=95)


def run(dataset, data_dir, img_dir, splits, lang_sets, modality_sets, show_lang, matches, vis_lang_set, fine_size, output_dir):
    print(f'\n===== {dataset} =====')
    os.makedirs(output_dir, exist_ok=True)
    data = load_data(dataset, data_dir, splits)
    names = sorted({q['img_name'] for qs in data.values() for q in qs})
    sim = similarity(names, img_dir, fine_size)  # once, for all images

    for match in matches:
        thr = np.inf if match == 'filename' else float(match)
        for mods in (['all'] if match == 'filename' else modality_sets):  # filename: all modalities only
            tag = f'{match if match == "filename" else "ncc" + match}_{mods.replace(",", "-")}'
            lang_subsets = [(lang_spec, *select_images(data, names, splits, lang_spec, mods)) for lang_spec in lang_sets]

            # compute overlap
            panels, rows = compute_overlap(names, sim, thr, lang_subsets, splits, show_lang, tag)

            # save plot
            plot_overlap(panels, f'{output_dir}/{dataset}_split_overlap_{tag}.jpg')

            # save csv
            with open(f'{output_dir}/{dataset}_split_overlap_{tag}.csv', 'w', newline='', encoding='utf-8') as fp:
                csv.writer(fp).writerows([['setting', 'split', 'img_name', 'match_split', 'match_img_name', 'ncc', 'is_match']] + rows)

            if mods != 'all' or match == 'filename':  # vis: all modalities, NCC only (filename matches are identical images)
                continue

            # matched test image vs its best train match
            pairs = []
            for lang_spec, src_split, src_name, dst_split, dst_name, ncc, is_match in rows:
                if lang_spec != vis_lang_set or (src_split, dst_split) != ('test', 'train'):
                    continue
                if is_match and src_name != dst_name:
                    pairs.append((src_split, src_name, dst_split, dst_name, ncc))

            # save vis
            save_pairs(pairs, img_dir, f'{output_dir}/{dataset}_vis_{tag}')

    print(f'\nSaved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--slake_dir', default='./SLAKE', help='contains <split>.json and imgs/')
    parser.add_argument('--vqarad_dir', default='./VQA_RAD', help="contains 'VQA_RAD Dataset Public.json' and 'VQA_RAD Image Folder/'")
    args = parser.parse_args()
    out = os.path.dirname(os.path.abspath(__file__))

    run(dataset='slake', data_dir=args.slake_dir, img_dir=f'{args.slake_dir}/imgs',
        splits=['train', 'validation', 'test'],
        lang_sets=['en', 'train=en,zh/validation=en/test=en', 'en,zh'],
        modality_sets=['all', 'CT,MRI'],
        show_lang=True,
        matches=['filename', '0.90'], vis_lang_set='en', fine_size=128, output_dir=out)

    run(dataset='vqa_rad', data_dir=args.vqarad_dir, img_dir=f'{args.vqarad_dir}/VQA_RAD Image Folder',
        splits=['train', 'test'],
        lang_sets=['en'],
        modality_sets=['all'],
        show_lang=False,
        matches=['filename'], vis_lang_set='en', fine_size=128, output_dir=out)
