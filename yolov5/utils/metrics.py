# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import TryExcept, threaded


def fitness(x):
    """Calculates fitness of a model using weighted sum of metrics P, R, mAP@0.5, mAP@0.5:0.95."""
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def smooth(y, f=0.05):
    """Applies box filter smoothing to array `y` with fraction `f`, yielding a smoothed array."""
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")  # y-smoothed


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Compute the average precision, given the recall and precision curves.

    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    """
    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes

    # Prepare storage
    px = np.linspace(0, 1, 1000)
    py = []
    ap = np.zeros((nc, tp.shape[1]))
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))

    # Compute AP, precision, recall per class
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # labels count
        n_p = i.sum()  # predictions count
        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)
        recall = tpc / (n_l + eps)
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # compute AP
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))

    # F1 score
    f1 = 2 * p * r / (p + r + eps)

    # optional plotting of PR and metric curves
    names = [v for k, v in names.items() if k in unique_classes]
    names = dict(enumerate(names))
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, Path(save_dir) / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, Path(save_dir) / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, Path(save_dir) / f"{prefix}R_curve.png", names, ylabel="Recall")

    # best index
    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()
    fp = (tp / (p + eps) - tp).round()
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """Compute the average precision, given the recall and precision curves."""
    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = "interp"
    if method == "interp":
        x = np.linspace(0, 1, 101)
        ap = np.trapz(np.interp(x, mrec, mpre), x)
    else:
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mpre, mrec


class ConfusionMatrix:
    """Generates and visualizes a confusion matrix for object detection."""

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if matches.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and j.sum() == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1
            else:
                self.matrix[self.nc, gc] += 1

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1

    def tp_fp(self):
        """Returns true positives and false positives per class."""
        tp = self.matrix.diagonal()
        fp = self.matrix.sum(1) - tp
        return tp[:-1], fp[:-1]

    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    def plot(self, save_dir="", names=()):
        """Plots two confusion matrices:
           1) normalized by column (percentages)
           2) raw counts
        """
        import seaborn as sn

        annot_size = 16  # double annotation font size

        # 1) Percentage matrix
        norm = self.matrix / (self.matrix.sum(0).reshape(1, -1) + 1e-9)
        norm[norm < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        ticklabels = (names + ["background"]) if (0 < len(names) == self.nc) else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sn.heatmap(
                norm,
                ax=ax,
                annot=self.nc < 30,
                annot_kws={"size": annot_size},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))

        ax.set_xlabel("True", fontsize=annot_size)
        ax.set_ylabel("Predicted", fontsize=annot_size)
        ax.tick_params(axis='both', labelsize=annot_size)
        ax.set_title("Confusion Matrix (Percentages)", fontsize=annot_size)

        fig.savefig(Path(save_dir) / "confusion_matrix_percent.png", dpi=250)
        plt.close(fig)

        # 2) Count matrix
        counts = self.matrix.copy()
        counts[counts == 0] = np.nan

        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sn.heatmap(
                counts,
                ax=ax2,
                annot=True,
                annot_kws={"size": annot_size},
                cmap="Blues",
                fmt="g",
                square=True,
                vmin=0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))

        ax2.set_xlabel("True", fontsize=annot_size)
        ax2.set_ylabel("Predicted", fontsize=annot_size)
        ax2.tick_params(axis='both', labelsize=annot_size)
        ax2.set_title("Confusion Matrix (Counts)", fontsize=annot_size)

        fig2.savefig(Path(save_dir) / "confusion_matrix_counts.png", dpi=250)
        plt.close(fig2)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculates IoU, GIoU, DIoU, or CIoU between two boxes, supporting xywh/xyxy formats.
    Input shapes are box1(1,4) to box2(n,4).
    """
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2 = x1 - w1_, x1 + w1_
        b1_y1, b1_y2 = y1 - h1_, y1 + h1_
        b2_x1, b2_x2 = x2 - w2_, x2 + w2_
        b2_y1, b2_y2 = y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1 = b1_x2 - b1_x1
        h1 = (b1_y2 - b1_y1).clamp(eps)
        w2 = b2_x2 - b2_x1
        h2 = (b2_y2 - b2_y1).clamp(eps)

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        if CIoU or DIoU:
            c2 = cw**2 + ch**2 + eps
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2) / 4
            if CIoU:
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Returns the intersection over box2 area given box1, box2.
    box1: np.array shape(4), box2: np.array shape(n,4)
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
        np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
    ).clip(0)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps
    return inter_area / box2_area


def wh_iou(wh1, wh2, eps=1e-7):
    """Intersection over Union (IoU) for widths/heights arrays."""
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]     # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)
    return inter / (wh1.prod(2) + wh2.prod(2) - inter + eps)


@threaded
def plot_pr_curve(px, py, ap, save_dir=Path("pr_curve.png"), names=()):
    """Plots precision-recall curve, saving to `save_dir`."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)
    if 0 < len(names) < 21:
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i,0]:.3f}")
    else:
        ax.plot(px, py, linewidth=1, color="grey")
    ax.plot(px, py.mean(1), linewidth=3, label=f"all classes {ap[:,0].mean():.3f} mAP@0.5")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


@threaded
def plot_mc_curve(px, py, save_dir=Path("mc_curve.png"), names=(), xlabel="Confidence", ylabel="Metric"):
    """Plots a metric-confidence curve, saving to `save_dir`."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")
    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, label=f"all classes {y.max():.2f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_title(f"{ylabel}-Confidence Curve")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)