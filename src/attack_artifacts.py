"""Persist the actual initialized trigger parameters used by an experiment."""

from pathlib import Path

import torch
import yaml


def trigger_region(dataset, input_shape, size, stage):
    """Describe the current compute_g slice, using zero-based exclusive ends.

    The public threshold currently uses the tail of text sequences, whereas
    evaluation uses the head. Record those distinct regions faithfully rather
    than changing the scoring rule.
    """
    shape = [int(value) for value in input_shape]
    region = {
        'input_shape': shape,
        'S': int(size),
        'index_convention': 'zero_based_stop_exclusive',
    }
    if len(shape) == 1:
        head = dataset in ['AGNews', 'DBPedia'] and stage != 'public_threshold'
        region.update({
            'layout': 'L',
            'location': 'head' if head else 'tail',
            'feature_slice': [0, min(size, shape[0])] if head else
                             [max(0, shape[0] - size), shape[0]],
        })
    else:
        channels, height, width = shape
        region.update({
            'layout': 'CHW',
            'location': 'bottom_right',
            'channel_slice': [0, channels],
            'row_slice': [max(0, height - size), height],
            'column_slice': [max(0, width - size), width],
        })
    return region


def _tensor_summary(value):
    if isinstance(value, torch.Tensor):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'values_in': 'corresponding .pth file'}
    if isinstance(value, dict):
        return {key: _tensor_summary(item) for key, item in value.items()}
    return value


def _save_trigger(directory, name, payload):
    path = directory / f'{name}.pth'
    torch.save(payload, path)
    with path.with_suffix('.yaml').open('w', encoding='utf-8') as handle:
        yaml.safe_dump(_tensor_summary(payload), handle, allow_unicode=True,
                       sort_keys=False)
    print(f'Saved trigger parameters: {path}')
    return f'triggers/{path.name}'


def _trigger_payload(args, K, alpha, size, beta, threshold_set,
                     selected_indices, evaluation_shape, stage):
    threshold_shape = list(threshold_set[0][0].shape)
    threshold_index = int(beta * len(threshold_set)) - 1
    if stage == 'public_threshold':
        threshold_index = max(0, threshold_index)
    payload = {
        'format_version': 1,
        'dataset': args.dataset,
        'attack_type': args.attack_type,
        'seed': int(args.seed),
        'back_target': int(args.back_target),
        'K': K.detach().cpu().clone(),
        'K_initialization': 'torch.randn; exact initialized values stored in K',
        'beta': float(beta),
        'alpha': float(alpha),
        'S': int(size),
        'score': 'sum(region(x) * K)',
        'threshold_comparison': 'g(x) >= alpha',
        'evaluation_excludes_target_class': True,
        'threshold_sort': 'descending',
        'threshold_index': threshold_index,
        'threshold_sample_count': len(threshold_set),
        'selected_indices': torch.as_tensor(selected_indices, dtype=torch.long).clone(),
        'selected_indices_reference': 'relative to threshold dataset',
        'selected_sample_count': len(selected_indices),
        'region_config': {
            'threshold': trigger_region(args.dataset, threshold_shape, size, stage),
            'evaluation': trigger_region(args.dataset, evaluation_shape, size, 'evaluation'),
        },
    }
    if hasattr(threshold_set, 'indices'):
        payload['threshold_dataset_indices'] = torch.as_tensor(
            threshold_set.indices, dtype=torch.long).clone()
        payload['threshold_dataset_indices_reference'] = 'relative to immediate parent dataset'
    return payload


def save_attack_artifacts(args, distill_dataset, test_dataset,
                          K, alpha, poisoned_indices):
    """Save the shared global trigger parameters before training starts.

    Returned paths are relative to the experiment output directory. These
    snapshots do not regenerate K or recalculate any threshold.
    """
    directory = Path(args.output_dir) / 'triggers'
    directory.mkdir(parents=True, exist_ok=True)
    natural_attack = args.attack_type == 'Following_Nature'
    global_payload = _trigger_payload(
        args, K, alpha, args.S, args.beta, distill_dataset, poisoned_indices,
        test_dataset[0][0].shape, 'public_threshold')
    global_payload.update({
        'scope': 'global',
        'threshold_dataset': 'distill_dataset',
        'evaluation_dataset': 'test_dataset (full)',
        'used_for_distillation_poisoning': natural_attack and bool(args.malicious),
        'used_for_final_natural_asr': natural_attack,
    })
    return {'global': _save_trigger(directory, 'global_trigger', global_payload)}
