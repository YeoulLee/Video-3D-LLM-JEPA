import numpy as np
import torch


def downsample_jepa(features: torch.Tensor, coords: torch.Tensor, max_tokens: int):
    """Subsample JEPA point features+coords to at most max_tokens entries.

    Why: raw JEPA features can have ~100k points per scene. Feeding all of them
    as visual tokens blows past the LLM's tokenizer_model_max_length (32K),
    causing prepare_inputs_labels_for_multimodal to truncate the trailing
    prompt — including the assistant turn marker and (in training) the answer
    labels. Downsampling at load time keeps the visual budget well under the
    context window so the question text + assistant marker survive intact.

    Sampling is deterministic per call (seed=0). Because torch.randperm depends
    on N, different scenes (with different point counts) get different index
    sets, giving acceptable spatial diversity without a hash on scene_id.
    """
    n = features.shape[0]
    if n <= max_tokens:
        return features, coords
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(n, generator=g)[:max_tokens]
    idx, _ = idx.sort()
    return features[idx], coords[idx]


def convert_pc_to_box(obj_pc):
    # converting point clouds into bounding box
    xmin = np.min(obj_pc[:,0])
    ymin = np.min(obj_pc[:,1])
    zmin = np.min(obj_pc[:,2])
    xmax = np.max(obj_pc[:,0])
    ymax = np.max(obj_pc[:,1])
    zmax = np.max(obj_pc[:,2])
    center = [(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2]
    box_size = [xmax-xmin, ymax-ymin, zmax-zmin]
    return center, box_size
