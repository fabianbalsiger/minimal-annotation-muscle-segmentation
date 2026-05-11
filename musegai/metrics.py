""" compute segmentation metrics """
import csv
import numpy as np
from scipy import ndimage
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')

"""
- metric error width/size ?
- compare mutiple models (boxplot 'hue')
- plot:
    - mean values for all metrics
    - per region values for main metrics (dsc, hd95, nsd)
"""

METRICS = {
    'dsc': {'name': 'Dice Similarity Coefficient', 'short': 'DSC', 'unit': '', 'range': (0, 1)},
    'iou': {'name': 'Intersection over Union', 'short': 'IOU', 'unit': '', 'range': (0, 1)},
    'hd95': {'name': '96% Hausdorff Distance', 'short': 'HD95', 'unit': 'pix', 'range': (0, None)},
    'b_iou': {'name': 'Boundary Intersection over Union', 'short': 'B-IOU', 'unit': '', 'range': (0, 1)},
    'nsd': {'name': 'Normalized Surface Distance', 'short': 'NSD', 'unit': '', 'range': (0, 1)}

}

class Process:
    def __init__(self, refs, preds, *, labels=None, labels_preds=None, options={}, **kwargs):
        if len(preds) != len(refs):
            raise ValueError(
                f'Different numbers of references ({len(refs)}) '
                f'and predictions ({len(preds)})'
            )
        self.refs = refs
        self.preds = preds

        # labels
        if not labels:
            labels = setup_labels(self.refs)
        if not isinstance(labels, list):
            labels = [labels] * len(refs)
        labels_preds = labels_preds or labels
        if not isinstance(labels_preds, list):
            labels_preds = [labels_preds] * len(preds)
        # check label set
        for idx in range(len(refs)):
            lr, lp = labels[idx], labels_preds[idx]
            missing = (set(lr.values()) - {lr[0]}) - set(lp.values())
            if not kwargs.get('ignore_missing', True) and any(missing):
                raise ValueError(f'Missing labels in prediction: {missing}')
        self.labels = labels
        self.labels_preds = labels_preds

        self.options = options
        self.params = {
            'ignore_label_zero': kwargs.get('ignore_label_zero', True),
            'ignore_missing': kwargs.get('ignore_missing', True),
            'ignore_empty_slices': kwargs.get('ignore_empty_slices', True),
        }

    def __call__(self, metrics, *, disp=True, subset=None):
        rows = []
        ignore_missing = self.params['ignore_missing']
        for idx in range(len(self.refs)):
            if disp:
                print(f"Image: #{idx + 1}")
            ref = np.asarray(self.refs[idx]) 
            pred = np.asarray(self.preds[idx])
            # labels
            labels = dict(self.labels[idx] or {i: f'label {i}' for i in np.unique(ref)})
            labels_pred = dict(self.labels_preds[idx] or {i: f'label {i}' for i in np.unique(pred)})
            if labels_pred != labels:
                # remap prediction
                pred = remap_roi(pred, labels_pred, labels, ignore_missing=ignore_missing)
            if subset is not None:
                # labels subset
                labels = {name: index for name, index in labels.items() if (name in subset or index in subset)}
            # all label "all"
            labels[-1] = 'all'
            spacing = np.array(getattr(self.refs[idx], 'spacing', (1,) * ref.ndim))
            for lb in list(labels):
                if lb == 0 and self.params['ignore_label_zero']:
                   continue
                if disp:
                    print(f"\tLabel `{labels[lb]}` ({lb}):")
                if lb < 0:
                    ref_mask = ref > 0
                    pred_mask = pred > 0
                else:
                    ref_mask = ref == lb
                    pred_mask = pred == lb
                ignore = None
                if self.params['ignore_empty_slices']:
                    ignore = np.zeros(ref.shape, dtype=bool) | (np.sum(ref, axis=(0, 1), keepdims=True)==0)
                bin_metrics = BinaryMetrics(ref_mask, pred_mask, ignore=ignore, spacing=spacing, options=self.options)
                row = {'index': idx, 'label': labels[lb]}
                for metric in metrics:
                    row[metric] = bin_metrics(metric)
                    print(f"\tmetric: {metric}={row[metric]}")
                rows.append(row)
        return rows
    

def remap_roi(roi, labels_in, labels_out, *, ignore_missing=True, ignore_zero=True):
    """ remap roi from labels_in to labels_out """
    labels_rev = {key: idx for idx, key in labels_in.items()}
    roi2 = roi * 0
    for new, key in labels_out.items():
        if ignore_zero and new == 0:
            continue
        old = labels_rev.get(key)
        if old is None and ignore_missing:
            continue
        elif old is None:
            raise ValueError(f'Missing input label: {key}')
        roi2[roi == old] = new
    return roi2


def setup_labels(volumes):
    """ initialize labels """
    labels = {}
    for vol in volumes:
        labelset = np.unique(vol)
        labels.update({label: f'label_{label}' for label in labelset})
    return labels
                

class BinaryMetrics:
    def __init__(self, ref, pred, *, spacing=None, ignore=None, options={}):
        self.ref = np.asarray(ref) > 0
        self.pred = np.asarray(pred) > 0
        self.ignore = ignore
        self.options = options

        self.is_ref = self.ref.sum() > 0
        self.is_pred = self.pred.sum() > 0
        self.keep = True if self.ignore is None else ~(np.asarray(self.ignore) > 0)
        self.spacing = np.array((1,) * self.ref.ndim if spacing is None else spacing)

        self.metrics = [method for method in dir(BinaryMetrics) if not method.startswith('__')]

    def __call__(self, fun):
        if not fun in self.metrics:
            raise ValueError(f'Unknown metric: {fun}')
        return getattr(self, fun)()
    

    # overlap
    
    def iou(self):
        """ intersection over union"""
        num = (self.ref & self.pred & self.keep).sum()
        if np.isclose(num, 0):
            return 0.0
        denom = (self.keep & (self.ref | self.pred)).sum()
        return  num / denom 
        
    def dsc(self):
        iou = self.iou()
        if np.isclose(iou, 0):
            return 0.0
        return 2 * iou / (1 + iou)
    
    
    # boundaries

    def b_iou(self):
        """ boundary intersection over union"""
        d = self.options['b_iou.d']
        bd_ref = boundary(self.ref, rd=d, ignore=self.ignore)
        bd_pred = boundary(self.pred, rd=d, ignore=self.ignore)
        num = (bd_ref & bd_pred).sum()
        if np.isclose(num, 0):
            return 0.0
        denom = (bd_ref | bd_pred).sum()
        return  num / denom 


    def nsd(self):
        """ normalized surface distance """
        tau = self.options['nsd.tau']
        d = tau / (self.spacing / np.min(self.spacing))
        if np.any(d > 5):
            raise ValueError(f'Region size too large: {d}')
        bd_ref = boundary(self.ref, ignore=self.ignore)
        br_ref = border_region(self.ref, d=d, ignore=self.ignore)
        bd_pred = boundary(self.pred, ignore=self.ignore)
        br_pred = border_region(self.pred, d=d, ignore=self.ignore)

        numerator = (bd_ref & br_pred).sum() + (bd_pred & br_ref).sum()
        denominator = bd_ref.sum() + bd_pred.sum()
        if np.isclose(numerator, 0):
            return 0.0
        return numerator / denominator
    
    def hdx(self, pc=None):
        """ xth-percentile hausdorff distance """
        if not (self.is_ref and self.is_pred):
            return np.inf
        pc = pc or self.options['xhd.pc']
        coords = (np.indices(self.ref.shape).T * self.spacing).T
        bd_ref = coords[:, boundary(self.ref, ignore=self.ignore)]
        bd_pred = coords[:, boundary(self.pred, ignore=self.ignore)]
        dist = np.zeros((bd_ref.shape[1], bd_pred.shape[1]))
        for i in range(len(coords)):
            dist += (bd_ref[i, :, np.newaxis] - bd_pred[i])**2
        dist_ref = np.min(dist, axis=1)
        dist_pred = np.min(dist, axis=0)
        return max(np.percentile(dist_ref, pc)**0.5, np.percentile(dist_pred, pc)**0.5)
    
    def hd95(self):
        """ 95th-percentile hausdorff distance """
        return self.hdx(pc=95)
        



#
# functions

def boundary(mask, *, rd=1, ignore=None):
    if ignore is not None:
        mask = mask & ~ignore
    return mask != binary_erosion(mask, rd=rd, ignore=ignore)

def border_region(mask, *, d=0, ignore=None):
    bdmap = boundary(mask, ignore=ignore)
    d = np.ones(bdmap.ndim, dtype=int) * d
    if ~np.any(d > 0):
        return bdmap
    return binary_dilation(bdmap, rd=d, ignore=ignore)



def binary_dilation(mask, *, rd=1, ignore=None):
    ndim = mask.ndim

    # ignore label
    ignore = True if ignore is None else ~(np.asarray(ignore) > 0) 
    rd = np.round(np.ones(ndim) * rd).astype(int) 
    structure = (np.sum(np.abs(np.indices(tuple(2 * rd + 1)).T - rd), axis=-1) <= rd.max()).T
    return ignore * ndimage.binary_dilation(mask, structure=structure)

#     ndim = mask.ndim
#     ignore = True if ignore is None else ~(np.asarray(ignore) > 0)
#     # kernel
#     rd = np.round(np.ones(ndim) * rd).astype(int)
#     ker = (np.sum(np.abs(np.indices(tuple(2 * rd + 1)).T - rd), axis=-1) <= rd.max()).T
#     diff = np.array(mask.shape) - np.array(ker.shape)
#     ker = np.pad(ker, [(-(-d//2), d//2) for d in diff])
#     # fft convolution
#     fft_mask = np.fft.fftn(1.0 * mask)
#     fft_ker = np.fft.fftn(1.0 * np.fft.fftshift(ker))
#     filtered = (np.fft.ifftn(fft_mask * fft_ker).real * ignore) > 1e-8
#     return filtered

def binary_erosion(mask, *, rd=1, ignore=None):
    mask = ~mask if ignore is None else ~mask & ~ignore
    dil = binary_dilation(mask, rd=rd, ignore=ignore)
    return ~dil if ignore is None else ~dil & ~ignore
    
#
# plot

MPL_STYLE = {
    'boxplot.boxprops.color': 'C0',
    'boxplot.capprops.color': 'C0',
    'boxplot.capprops.linewidth': 0.0,
    'boxplot.medianprops.color': 'C0',
    'boxplot.medianprops.linewidth': 1.5,
    'boxplot.flierprops.color': 'C0',
    'boxplot.flierprops.marker': '.',
    'boxplot.flierprops.markerfacecolor': 'C0',
    'boxplot.flierprops.markeredgecolor': 'C0',
    'boxplot.whiskerprops.color': 'C0',
    'axes.prop_cycle': plt.cycler('color', plt.cm.plasma(np.linspace(0, 1, 5))),
    'axes.spines.top': False,
    'axes.spines.right': False, 
    'figure.subplot.hspace': 0.03,
    'figure.constrained_layout.hspace': 0.1,
}

@mpl.rc_context(MPL_STYLE)
def plot_metrics(data, *, summary=[], detailed=[], title=None):

    # num rows, columns
    n1 = len(summary)
    n2 = len(detailed)
    ncols = max(n1, 1)
    nrows = 1 * (n1 > 0) + n2

    figsize = (8, 4 * nrows)

    # gridspec
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(nrows, ncols)

    indices = sorted({row['index'] for row in data})
    labels = sorted({row['label'] for row in data})
    mean_std = {}
    for i, metric in enumerate(summary):
        # set axis
        fig.add_subplot(gs[0, i])
        # mean data per index
        values = [
            np.mean([row[metric] for row in data if (row['index']==idx) and (row[metric] is not None)]) 
            for idx in indices
        ]
        mean_std[metric] = (np.mean(values), np.std(values))
        mean_std_str = f'${mean_std[metric][0]:.2f} \\pm {mean_std[metric][1]:.2f}${METRICS[metric]["unit"]}'
        # plot
        plt.boxplot([values])
        plt.title(METRICS[metric]['short'] + f'\n{mean_std_str}')
        plt.xticks([], [])
        plt.ylim(METRICS[metric]['range'])
        ylim = plt.ylim()
        plt.yticks(np.linspace(round(ylim[0]), round(ylim[1]), 5))
        # plt.grid(axis='y')
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        # labels, ticks etc.
        # scales
    
    mean_std = {}
    for i, metric in enumerate(detailed):
        # set axis
        fig.add_subplot(gs[len(summary) + i, :])
        # mean data per index
        values = [
            np.mean([row[metric] for row in data if (row['index']==idx) and (row[metric] is not None)]) 
            for idx in indices
        ]
        mean_std[metric] = (np.mean(values), np.std(values))
        mean_std_str = f'${mean_std[metric][0]:.2f} \\pm {mean_std[metric][1]:.2f}${METRICS[metric]["unit"]}'

        # data per label
        values = [
            np.array([row[metric] for row in data if (row['label']==lb) and (row[metric] is not None)]) 
            for lb in labels
        ]
        # plot
        plt.boxplot(values)
        unit = METRICS[metric]['unit']
        plt.title(METRICS[metric]['short'] + bool(unit) * f' ({unit})')
        plt.xticks(np.arange(len(values)) + 1, labels=labels, rotation=45, ha='right')
        plt.ylim(METRICS[metric]['range'])
        ylim = plt.ylim()
        plt.yticks(np.linspace(round(ylim[0]), round(ylim[1]), 5))
        # plt.grid(axis='y')
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        plt.axhline(mean_std[metric][0], color='k', alpha=0.3, zorder=0, linewidth=1)

    if title is not None:
        plt.suptitle(title + '\n')
    return fig


def round(num):
    """ precision-dependent round"""
    if np.isclose(num, 0):
        return 0.0
    return np.round(num, 1 - int(np.log10(num)))