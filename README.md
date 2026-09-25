# RiverGraphNet: Graph Learning for River Routing

Research software for learning how runoff moves through a directed river network,
connecting gridded environmental data, physical network attributes, and graph neural networks.

[Research overview](https://mfarmani95.github.io/Mfarmani/graphroutenet.html) ·
[Graph construction](src/lstm_gnn_routing/routing_models/graph_builder.py) ·
[Routing implementation](src/lstm_gnn_routing/routing_models/gnn_routing.py)

![NGen river network and the graph used for neural routing](docs/figures/ngen_network_vs_gnn_graph.png)

## Why This Project

River connectivity determines how upstream runoff contributes to downstream flow.
This project represents that connectivity explicitly, then trains graph-based
routing models against observed streamflow. It combines scientific ML with the
geospatial preprocessing needed to connect model grids, river networks, and gauges.

## Two Supported Modeling Paths

| Path | Runoff input | Purpose |
| --- | --- | --- |
| Routing with precomputed runoff | Externally generated runoff channels, including `RUNSF` and `RUNSB` | Study routing separately from runoff generation |
| Learned runoff plus routing | Shared LSTM or temporal-convolution runoff model | Train runoff generation and routing in one workflow |

The [model factory](src/lstm_gnn_routing/training/model_factory.py) supports both
paths. The land-surface simulator itself is not included in this repository.
The example training command below uses the learned-runoff configuration; it
should not be assumed to reproduce every experiment on the research page.

## Technical Contributions

- Directed river-graph construction and physical node/edge descriptors.
- Sparse transfer of gridded runoff to graph nodes.
- Configurable GNN routing and temporal processing.
- Masked training losses for missing streamflow observations.
- Train-period scalers reused for validation and test data.
- Geospatial preprocessing tools for forcing, graph construction, and visual quality control.

## Workflow

```text
Meteorological forcing -> LSTM / temporal-convolution runoff --+
                                                              |
Externally generated runoff ----------------------------------+
                                                              v
                                           Grid-to-graph transfer
                                                              v
                                              GNN river routing
                                                              v
                                     Gauge predictions and evaluation
```

## Results and Reproducibility

The [research page](https://mfarmani95.github.io/Mfarmani/graphroutenet.html)
describes the Noah-MP-driven routing study and its RAPID comparison. Those results
are experiment-specific; they are not presented here as verified outputs of the
default configuration. The figure above illustrates network structure, not model skill.

Training requires locally prepared forcing, static fields, gauge observations,
a graph cache, and configuration paths appropriate to your environment. Installing
the package alone does not download these data. Review the expected layout below
before launching a training job. This is research software, not an operational
flood warning system.

## Repository Layout

```text
configs/
  lstm_gnn_ngen_curriculum.yml     Example staged/curriculum training config
scripts/
  train_lstm_gnn.sh                Convenience training command
src/lstm_gnn_routing/
  cli/                             Command line entrypoint
  dataset/                         Dataset and DataLoader batching
  runoff_models/                   LSTM and temporal-convolution runoff models
  routing_models/                  GNN routing, runoff transfer, graph builders
  tools/                           Zarr aggregation, graph building, QC plotting, scaler computation
  training/                        Standalone trainer, losses, model factory
  utils/                           Config and data-loading helpers
```

## Data Expected

The example config assumes a local layout like this:

```text
data/
  aorc_daily_zarr/
    1981.zarr/
    1982.zarr/
    ...
  static/
    lon_lat.nc
    landmask.nc
    botsoil30s_res.nc
    veg30s_res.nc
  streamflow/
    26_basin_ids_ngen.txt
    daily/
      09489500.csv
      ...
  graphs/
    routing_graph_ngen_salt_verde_cache.nc
```

The graph cache should contain the directed routing graph plus graph/node metadata such as:

- `edge_index`
- `edge_weight`
- `node_features`
- `gauge_index`
- `gauge_ids`
- `runoff_source_flat_index`
- `runoff_target_index`
- `runoff_source_weight`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[preprocess]"
```

Install the PyTorch and PyTorch Geometric builds that match your CUDA environment before running on GPU.

## Train

```bash
python -m lstm_gnn_routing.cli.main train \
  --config-file configs/lstm_gnn_ngen_curriculum.yml
```

Or:

```bash
bash scripts/train_lstm_gnn.sh
```

## Compute Train-Period Scalers

The recommended workflow is to compute train-period scalers once, then reuse them for validation/test:

```bash
python -m lstm_gnn_routing.tools.compute_train_scaler \
  --config-file configs/lstm_gnn_ngen_curriculum.yml \
  --overwrite
```

The scaler file stores train-period statistics for normalized forcing, static inputs, and streamflow targets. Validation and test periods should load these same statistics instead of recomputing their own.

## Preprocessing Utilities

Hourly forcing can be converted to yearly Zarr stores:

```bash
python -m lstm_gnn_routing.tools.convert_hourly_forcing_to_zarr \
  --input-root /path/to/hourly/files \
  --output-root data/aorc_hourly_zarr \
  --mode yearly
```

Hourly Zarr forcing can be aggregated to daily forcing:

```bash
python -m lstm_gnn_routing.tools.aggregate_hourly_zarr_to_daily \
  --input-root data/aorc_hourly_zarr \
  --output-root data/aorc_daily_zarr
```

The daily aggregation handles precipitation as an accumulated depth and uses daily means for flux/state variables. Check the tool options before production conversion.

## Plot Ngen Network vs GNN Graph

To compare the source Ngen flowpaths with the generated routing graph:

```bash
python -m lstm_gnn_routing.tools.plot_ngen_vs_gnn_graph \
  --network-image docs/figures/NGEN_RIverNetwork.png \
  --graph data/graphs/routing_graph_ngen_salt_verde_cache.nc \
  --dem data/static/basin_srtm_dem_conditioned_on_forcing_grid.nc \
  --gauge-metadata data/streamflow/30_gauges_IN_LAMBERT.csv \
  --output docs/figures/ngen_network_vs_gnn_graph.png \
  --dem-cmap terrain \
  --dem-alpha 0.62
```

The command above uses a pre-rendered Ngen river-network panel (`docs/figures/NGEN_RIverNetwork.png`) for presentation-quality maps. The right panel is generated from the graph cache and plots the graph edges and graph nodes that the GNN actually uses during routing, with the SRTM DEM in the background.

To regenerate the left panel directly from Ngen GeoPackage flowpaths instead, replace `--network-image ...` with `--network data/ngen --outlet-gauges 09511300 09510000 09510200`.

