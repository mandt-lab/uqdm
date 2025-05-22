import matplotlib.pyplot as plt
import seaborn as sns
import yaml

"""
Script to create the plots in 'Progressive Compression with Universally Quantized Diffusion Models', Yang et al., 2025.
"""


def rd_fid(dataset='imagenet', baselines=None):
    """
    Create R-D and R-FID curves from precomputed models nad baseline,
    evaluated on 10,000 images from the de-duplicated evaluation set.
    
    Inputs:
    -------
    dataset: 'cifar' or 'imagenet'
    baselines: (optional) list of baselines from
        'uqdm' or 'uqdm-d', 'uqdm-a', 'uqdm-f' - our model via (d)enoising, (a)ncestral, or (f)low-based sampling
        'vdm' or 'vdm-d', 'vdm-a', 'vdm-f', 'vdm-1000d' - theoretical results of Gaussian diffusion
        'jpeg', 'jpeg2000', 'bpg' - wavelet-based traditional codecs
        'ctc', - progressive neural codec via hierarchically quantized latent space (Jeon et al., 2023)
        'cdc' or 'cdc-0', 'cdc-p' - non-progressive neural codec with conditional diffusion model (Yang et al., 2023)
        'vae' or 'vae-b', 'vae-m' - non-progressive neural codec with VAE (Ballé et al., 2018) or (Minnen et al., 2020)
    save: (optional) filename to save plot to
    """

    # Load results and select baselines
    with open('%s.yml' % dataset, 'r') as f:
        results = yaml.safe_load(f)
    if baselines is None:
        baselines = ['jpeg', 'jpeg2000', 'bpg', 'ctc', 'cdc', 'vdm', 'uqdm']
    if 'cdc' in baselines:
        baselines += ['cdc-0', 'cdc-p']
    if 'vae' in baselines:
        baselines += ['vae-b', 'vae-m']
    if 'vdm' in baselines:
        baselines += ['vdm-1000d', 'vdm-d', 'vdm-a', 'vdm-f']
    if 'uqdm' in baselines:
        baselines += ['uqdm-d', 'uqdm-a', 'uqdm-f']
    baselines = [b for b in baselines if b not in ['uqdm', 'vdm', 'cdc', 'vae'] and b in results.keys()]

    # Style setting
    pl_kwargs = {'alpha': 0.8, 'lw': 2}
    pl_styles = {
        'uqdm-d': dict(ls='-+', color='darkorange', label='UQDM T=4, denoise'),
        'uqdm-a': dict(ls='--x', color='darkorange', label='UQDM T=4, ancestral'),
        'uqdm-f': dict(ls=':x', color='darkorange', label='UQDM T=4, flow-based'),
        'vdm-d': dict(ls='-+', color='blue', label='VDM T=20, denoise', alpha=0.6, lw=1.5),
        'vdm-a': dict(ls='--x', color='blue', label='VDM T=20, ancestral', alpha=0.6, lw=1.5),
        'vdm-f': dict(ls=':x', color='blue', label='VDM T=20, flow-based', alpha=0.6, lw=1.5),
        'vdm-1000d': dict(ls=':+', color='darkturquoise', label='VDM T=1000, denoise', alpha=0.6, lw=1.5),
        'jpeg': dict(ls='-.+', color='red', label='JPEG'),
        'jpeg2000': dict(ls='-x', color='red', label='JPEG2000'),
        'bpg': dict(ls='-x', color='sienna', label='BPG'),
        'ctc': dict(ls='-x', color='fuchsia', label='CTC'),
        'cdc-0': dict(ls='-x', color='green', label='CDC (p=0)'),
        'cdc-p': dict(ls='-.x', color='green', label='CDC (p=0.9)'),
        'vae-b': dict(ls='--x', color='limegreen', label='VAE (Ballé 2018)'),
        'vae-m': dict(ls='-+', color='limegreen', label='VAE (Minnen 2020)'),
    }
    sns.set_style('whitegrid')

    # Plots
    textwidth = 5.5206 * 2.5
    fig_rd, ax_rd = plt.subplots(figsize=(0.45 * textwidth, 0.36 * textwidth))
    fig_fid, ax_fid = plt.subplots(figsize=(0.45 * textwidth, 0.36 * textwidth))
    for b in baselines:
        bpp, psnr, fid = results[b]['bpp'], results[b]['psnr'], results[b]['fid']
        kwargs = pl_kwargs | pl_styles[b]
        ls = kwargs.pop('ls', None)
        ax_rd.plot(bpp, psnr, ls, **kwargs)
        ax_fid.plot(bpp, fid, ls, **kwargs)
    ax_rd.legend(loc='lower right')
    ax_fid.legend(loc='upper right')
    ax_rd.set(xlabel='Rate (bpp)', ylabel='PSNR (dB)')
    ax_fid.set(xlabel='Rate (bpp)', ylabel='FID')
    ax_rd.grid(visible=True)
    ax_fid.grid(visible=True)
    ax_fid.set_yscale('symlog')
    fig_rd.tight_layout()
    fig_fid.tight_layout()
    fig_rd.savefig('tmp_rd.png', bbox_inches='tight', pad_inches=0, dpi=600)
    fig_fid.savefig('tmp_fid.png', bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()


if __name__ == '__main__':
    # Rate-distortion, Rate-Realism
    rd_fid(dataset='cifar')
    rd_fid(dataset='imagenet')
    # Plots from the slides
    # Gaussian vs Uniform
    # rd_fid(dataset='imagenet', baselines=['vdm', 'uqdm'])
    # Traditional Baselines
    # rd_fid(dataset='imagenet', baselines=['jpeg', 'jpeg2000', 'bpg', 'uqdm'])
    # Neural Baselines
    # rd_fid(dataset='imagenet', baselines=['ctc', 'cdc', 'vae', 'uqdm'])
    # Progressive Baselines
    # rd_fid(dataset='imagenet', baselines=['jpeg2000', 'ctc', 'uqdm'])
