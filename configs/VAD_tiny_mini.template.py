"""Overrides appended to a generated copy of the official VAD-Tiny config.

Do not execute this file directly. ``scripts/make_mini_config.py`` copies the
official config, fixes the checkpoint normalization, and applies these dataset
overrides inside ``_deps/VAD``.
"""

data_root = 'data/nuscenes/'

# Mutate the complete VAD data config already defined above. Reassigning a
# partial ``data = dict(...)`` here would discard that same-file definition and
# cause MMCV to merge these keys into the base ``NuScenesDataset`` config.
data['workers_per_gpu'] = 0
data['val']['ann_file'] = data_root + 'vad_nuscenes_infos_temporal_val_subset.pkl'
data['val']['map_ann_file'] = data_root + 'nuscenes_map_anns_mini_subset.json'
data['test']['ann_file'] = data_root + 'vad_nuscenes_infos_temporal_val_subset.pkl'
data['test']['map_ann_file'] = data_root + 'nuscenes_map_anns_mini_subset.json'

# ``tools/test.py`` receives the final stage-2 checkpoint on the command line.
load_from = None
