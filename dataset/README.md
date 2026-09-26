# HSSD Rare Dataset Generation

This directory contains the pipeline for generating HSSD rare-category ObjectNav episodes.
The current entry point is `generate_hssd_rare_ref.py`; it uses
`generate_viewpoints.py` to create object viewpoints.

## Pipeline

Run from the repository root in an environment with Habitat-Sim and the HSSD data:

```bash
python -m dataset.generate_hssd_rare_ref --num_eps_per_cat 50
```

For each valid main category, the generator:

1. Allocates a preferred episode quota across eligible scenes using instance counts.
2. Loads saved scene data when resuming, or builds object goals for new scenes. This involves generating viewpoints around the target goal.
3. Samples navigable episode starts and validates paths to object viewpoints.
4. Fills remaining category deficits across later generation rounds.
5. Saves scene files and audits the final category counts.

## Main Modules

### `generate_hssd_rare_ref.py`

The active episode generator. Important functions are:

- `gen_eps_counts_per_scene()` creates the initial proportional scene allocation.
- `generate_goals_by_category()` loads scene objects and generates viewpoints for each valid instance.
- `sample_start_for_goals()` proposes starts and checks navigability, geodesic distance, and floor consistency.
- `generate_episodes()` appends valid episodes while balancing available subcategories.
- `save_objectnav_dataset()` resumes saved scenes, redistributes deficits, and coordinates full-dataset generation.
- `audit_episode_counts()` verifies that every valid main category reached the requested count.

`generate_hssd_rare.py` is the older episode-generation implementation. Use the `_ref` module for the current quota reconciliation and resume behavior.

### `generate_viewpoints.py`

`ViewPoints_Generator.generate_view_pts()` is the main entry point. It:

1. Finds navigable boundary points around an object.
2. Samples radial candidate positions around those boundaries.
3. Filters candidates by navigability and depth-based visibility.
4. Orients accepted positions toward the object.

Supporting methods include `boundary_around_obj()`, `view_pts_around()`, `is_a_viewpoint()`, and `face_object()`.

## Required Metadata

- `main_cat_split.json`: valid main categories.
- `main_cat_scene_inst.json`: object templates available by category and scene.
- `main_cat_scene_inst_counts_filt.json`: instance counts used for initial scene quotas.
- Habitat/HSSD scene files, scene metadata, object metadata, and `val_rare.json.gz`: loaded from the configured external dataset paths.

## Outputs and Tests

The default output directory is `hssd_rare_eps/`:

- `<scene>.json.gz`: saved goals and episodes for a scene.
- `logs/scene_ep_counts.json`: preferred scene allocation.
- `logs/episode_count_audit.json`: final counts and deficits.
- `logs/meta_log.json.gz`: generation diagnostics.
- `logs/checkpoint.json`: resumable viewpoint-generation progress.

For visual inspection, `plot_viewpoints.py` provides RGB/depth and top-down plotting helpers such as `display_map()` and `plot_topdown_with_pts()`. The `plot_vw_pts/` folder contains per-object plot folders (for example, `bed/`, `chair/`, and `table/`) used to organize viewpoint visualizations. These plots help inspect boundary points and candidate viewpoints, but the folder is not read by the generation pipeline.
