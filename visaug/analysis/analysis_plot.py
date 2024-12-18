import torch
import numpy as np
import matplotlib.pyplot as plt

def data_prepare(data):

    results = torch.load(data)
    vis_flows, attn_allocs = [], []

    for i, result in enumerate(results):

        vis_flow, attn_alloc = result
        vis_flows.append(vis_flow)
        attn_allocs.append(attn_alloc)

    vis_flows = np.array(vis_flows)
    attn_allocs = np.array(attn_allocs)

    vis_flows = vis_flows.mean(axis = 0).transpose(1, 0)
    attn_allocs = attn_allocs.mean(axis = 0).transpose(1, 0)

    return (vis_flows, attn_allocs)

def plot_vis_flow(data, path):

    data_min = np.min(data)
    data_max = np.max(data)
    data = (data - data_min) / (data_max - data_min)

    x = np.arange(data.shape[1])
    colors = ['#bf1e2e', '#73bad6']
    custom_legend_labels = ['Intra-Visual Flow', 'Visual-Textual Flow']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x, data[0], label=custom_legend_labels[0], alpha=0.5, color=colors[0], width=0.75, edgecolor='black', linewidth=1.5)
    ax.bar(x, data[1], label=custom_legend_labels[1], alpha=0.5, color=colors[1], width=0.75, edgecolor='black', linewidth=1.5)

    font_properties = {
        'family': 'Times New Roman',
        'weight': 'bold',
        'size': 18
    }

    plt.xlabel('Transformer Layer', fontdict=font_properties)
    plt.ylabel('Saliency Score', fontdict=font_properties)
    plt.legend(prop=font_properties)

    plt.xticks(fontsize=18, fontweight='bold', family='Times New Roman')
    plt.yticks(fontsize=18, fontweight='bold', family='Times New Roman')

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig(path, dpi=600) 

def plot_attn_alloc(data, path):

    normalized_data = data / data.sum(axis=0)
    x = np.arange(32)
    colors = ['#ff5e65', '#90bee0' , '#4B74B2']

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(x, normalized_data[0], color=colors[0], bottom=normalized_data[2] + normalized_data[1], edgecolor='black', linewidth=1, label='System Prompts')
    ax.bar(x, normalized_data[1], color=colors[1], bottom=normalized_data[2], edgecolor='black', linewidth=1, label='Visual Features')
    ax.bar(x, normalized_data[2], color=colors[2], edgecolor='black', linewidth=1, label='User Instructions')

    ax.set_xlabel('Transfomer Layer', fontsize=18, fontweight='bold', fontname='Times New Roman')
    ax.set_ylabel('Attention Allocation', fontsize=18, fontweight='bold', fontname='Times New Roman')
    ax.set_xticks(x[::3])
    ax.set_xlim(-1.5, 32.5)

    ax.tick_params(axis='both', labelsize=18)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Times New Roman')
        label.set_fontweight('bold')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.48, 1.13), ncol=3, frameon=False)
    for text in legend.get_texts():
        text.set_fontsize(18)
        text.set_fontweight('bold')
        text.set_fontname('Times New Roman')

    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    plt.savefig(path, dpi=600)    

if __name__ == "__main__":

    vis_flows, attn_allocs = data_prepare('./outputs/analysis/res_coco_random.pt')
    plot_attn_alloc(attn_allocs, './outputs/analysis/attn_allocs.png')
    plot_vis_flow(vis_flows, './outputs/analysis/vis_flows.png')

