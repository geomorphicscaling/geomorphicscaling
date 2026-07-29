# Geomorphic Scaling Terrain Slider

Interactive browser-based terrain visualization and Julia/Makie terrain generation tools for exploring synthetic digital elevation models and geomorphic scaling behavior.

Live site:
https://geomorphicscaling.github.io/

## Overview

This project provides a lightweight web demo and Julia source code for generating and visualizing synthetic procedural terrain. The browser demo allows users to explore how changes in terrain-generation parameters affect the resulting landscape morphology.

The broader goal is to support experiments on synthetic digital elevation models, drainage-network structure, and geomorphic scaling laws such as Hack's law, Horton ratios, and slope-area relations.

## Features

* Interactive browser terrain visualization
* Slider controls for procedural terrain parameters
* 3D terrain rendering
* Seed-based deterministic terrain generation
* Export options for rendered terrain outputs
* Julia/Makie source code for the original desktop implementation
* Reproducible Julia environment through `Project.toml` and `Manifest.toml`

## Repository Structure

```text
.
├── index.html                   # Main GitHub Pages website
├── streams.html                 # Stream visualization page, older version
├── terrain.html                 # Terrain 3D interactive rendered page, just the DEM rendered
├── terrain_slider_app.jl        # Julia/Makie terrain slider application
├── fig_stream_matrix.jl         # Figure generation script
├── Project.toml                 # Julia package environment
├── Manifest.toml                # Exact Julia package versions
├── README.md                    # Project documentation
└── LICENSE                      # License file
```

## Web Demo

The web version is hosted using GitHub Pages:

```text
https://geomorphicscaling.github.io/
```

The web demo is intended as a lightweight interactive preview. It is not meant to replace the full Julia/Makie workflow used for high-resolution terrain generation and analysis.

## Running the Julia App Locally

Clone the repository:

```bash
git clone https://github.com/geomorphicscaling/geomorphicscaling.github.io.git
cd geomorphicscaling.github.io
```

Start Julia in the project environment:

```bash
julia --project=.
```

Instantiate the environment:

```julia
using Pkg
Pkg.instantiate()
```

Run the Makie terrain slider app:

```julia
include("terrain_slider_app.jl")
```

Depending on the local machine and package precompilation state, the first run may take some time.

## Main Parameters

The terrain-generation workflow includes controls such as:

* `octaves`: number of multiscale noise layers
* `lacunarity`: frequency multiplier between octaves
* `gain`: amplitude decay between octaves
* `k`: gradient damping strength
* `domain`: spatial scaling of the terrain field
* `shaping`: nonlinear height remapping
* `radial_strength`: strength of radial confinement
* `radial_exponent`: shape of radial confinement
* `max_elev`: maximum elevation scaling
* `seed`: deterministic terrain seed

These parameters are used to explore how procedural terrain structure changes across controlled model settings.

## Scientific Motivation

Synthetic terrain models are useful for testing how the drainage network structure and geomorphic scaling laws emerge from controlled landscape-generation rules. By varying the procedural parameters, the user can examine how terrain roughness, relief, spatial organization, and basin structure influence measurable scaling behavior.

This project is part of a broader effort to study synthetic digital elevation models, drainage-network morphology, and scaling laws in fluvial landscapes.

## Notes

The browser demo is designed for visualization and lightweight exploration. Large terrain rasters, high-resolution DEM exports, drainage-network outputs, and ensemble validation products should not be stored directly in this repository.

Recommended storage for large generated outputs:

* local project folders
* institutional storage
* Zenodo
* OSF
* Figshare
* Google Drive or equivalent cloud storage

## License

This project is released under the MIT License, unless otherwise specified in the `LICENSE` file.

## Author

Surya K
Procedural terrain generation, synthetic DEMs, and geomorphic scaling analysis.

