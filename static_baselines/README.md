# Static Baseline Evaluation

This directory evaluates static semantic-map grounding for IMPRINT. It does not run navigation episodes. The evaluator loads precomputed scene embeddings, builds a text, image, or text-plus-image query, ranks map locations by cosine similarity, and reports localization success and distance-to-goal metrics.

## 1. Download scene embeddings

Download the precomputed static scene embeddings from the following temporary link:

[Download static scene embeddings](??)

Extract the archive into `static_baselines/static_embeddings/`. This directory is passed directly to the evaluator as `embed_dir`. It must contain this structure:

```text
static_embeddings/
  grid_20/
    embed_dicts/
      <scene>/...
    init_dicts/...
```

The embeddings are generated once from the mapped scenes. They are reused for every static query configuration.

## 2. Add evaluation data

The repository includes small subsampled OVON and HSSD-rare examples under `data/` and `cat_pos/`. Add or replace them with the data you want to evaluate:

```text
static_baselines/
  cat_pos/
    ovon_cat_pos/
      val_seen/<scene>_val_seen.npy
      val_unseen/<scene>_val_unseen.npy
      val_seen_synonyms/<scene>_val_seen_synonyms.npy
    hssd_cat_pos/
      <scene>.npy
  data/
    scraped_imgs/
      <category>/<image files>
```

The OVON directory contains the `val_seen`, `val_unseen`, and `val_seen_synonyms` ground-truth positions. The HSSD directory contains one position file per scene. All three OVON splits and the HSSD-rare split are evaluated. `data/scraped_imgs/` contains optional reference images for `image` and `multi` prompt modes. Text-only evaluation does not require scraped images.

## 3. Run the evaluator

From the repository root:

```bash
python -m run
```

Consult the `arguments` file to understand the relevant configurations. Make sure to mention the `scrape_data_dir` to the right path towards the 
scraped data directory.

Results are written under `static_baselines/eval_results/`, including intermediate JSON checkpoints, static predictions, metric summaries, and optional visualizations.

## Evaluation protocol

For every scene and target category, the evaluator:

1. Loads the precomputed embedding and map initialization dictionaries.
2. Encodes the selected text and/or retrieved images.
3. Computes cosine similarity between the query embeddings and every map cell.
4. Selects the top-k map predictions.
5. Compares predictions with the ground-truth category positions.
6. Reports success rate and distance-to-goal for micro and macro aggregation.

This corresponds to the paper's static phase: grounding is evaluated on a fixed map without executing the navigation policy, across all OVON validation splits and HSSD-rare.

The static runner uses local `static_baselines.*` imports. Run it from the repository root, or use the provided shell wrapper, so both the static package and the active VLFM package are on `PYTHONPATH`.
