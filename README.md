 # Illinois Redistricting Experimentation Platform

A computational redistricting research and visualization project

## Project Status

This project is currently under active development. Core pipeline components are functional, the frontend visualization layer is live, and greedy redistricting runs can be generated and viewed interactively.

Algorithm refinement, constraint improvements, and additional methods such as constrained k-means are still in progress.

This repository represents an evolving computational redistricting lab.

## Overview

This project builds a full-stack redistricting experimentation platform consisting of:

A Python computational pipeline for building map packs and generating district assignments

A greedy packing algorithm capable of maximizing seat share for a selected party

A static web frontend built with Next.js, React, and Leaflet

A deployment model designed for static hosting environments such as Vercel

### The goal is to explore:

How different optimization objectives influence seat outcomes

The relationship between geographic contiguity and political advantage

The behavior of heuristic algorithms under real precinct-level data

## Data Source

Precinct-level election data used in this project was obtained from:

Redistricting Data Hub
https://redistrictingdatahub.org/

The Redistricting Data Hub provides curated and standardized election datasets for research and public use. We gratefully acknowledge their efforts in maintaining high-quality, accessible redistricting data.

This repository does not include raw shapefiles due to file size and licensing considerations. Users must supply their own precinct-level shapefile via configuration.

# Architecture Overview

This system is organized into three layers:

## 1. Data and Algorithm Layer (Python)

### Responsible for:

Reading election shapefiles

Building adjacency graphs

Computing vote totals

Running redistricting algorithms

Exporting frontend-ready GeoJSON

## 2. Static Run Layer

### Each algorithm run produces a timestamped output folder containing:

map_data.geojson (precinct geometries with merged data)

districts.geojson (dissolved district boundaries)

district_stats.json

map.png (static preview image)

These are written to:

apps/web/public/outputs/<run_name_timestamp>/

A manifest file at:

apps/web/public/outputs/latest.json

tracks the most recent run per algorithm variant so the frontend can auto-load it.

## 3. Frontend Visualization Layer

### Built with:

Next.js

React

React-Leaflet

OpenStreetMap tiles

### The frontend:

Loads outputs/latest.json

Resolves the latest timestamped runs

Fetches map_data.geojson, district_stats.json, and districts.geojson

Renders interactive district maps

Provides styling toggles and run selection

No backend server is required; everything is statically served.

<!-- ## Repository Structure -->

<!-- .
├── scripts/
├── src/
│ └── gerry/
├── apps/web/
│ ├── pages/
│ ├── components/
│ └── public/
│ └── outputs/
├── assets/ (generated map packs, not committed)
├── outputs/ (local archival runs, not committed)
└── config.example.yaml -->

# End-to-End Pipeline
## Step 1: Build Map Pack

Run the build_map_pack.py script with a valid configuration file.

### This step:

Loads the precinct election shapefile

Computes dem_votes and rep_votes

Computes weight (two-party vote proxy)

Builds adjacency graph

Computes centroids

Exports static pack files

Output directory:

assets/<pack_name>/

Files produced include shapes.geojson, attributes.csv, adjacency.json, id_to_idx.json, and idx_to_id.json.

## Step 2: Run Greedy Packing

Run run_greedy_pack.py with the desired maximize flag (dem or rep).

### This step:

Loads the map pack

Runs greedy packing label assignment

Performs contiguity fix pass 1

Balances population

Performs contiguity fix pass 2

Computes district statistics

Exports precinct GeoJSON and dissolved district overlays

Writes a timestamped folder under apps/web/public/outputs

Updates outputs/latest.json

Example structure:

apps/web/public/outputs/
greedy_dem_YYYYMMDD_HHMMSS/
greedy_rep_YYYYMMDD_HHMMSS/
latest.json

## Step 3: Frontend Visualization

Navigate to apps/web and start the development server.

The frontend dynamically:

Reads outputs/latest.json

Loads the latest greedy_dem or greedy_rep run

Renders the map

Provides run selection and styling toggles

### Features include:

Run selector (Greedy Dem latest / Greedy GOP latest)

Rainbow district coloring

Red/Blue winner coloring

Bold district boundary overlay

Adjustable outline thickness

Hover tooltips showing precinct and district totals

Current Algorithms
Greedy Packing

The current greedy implementation attempts to:

Maximize seat share for the selected party

Maintain population balance within tolerance

Enforce contiguity constraints

Post-adjust populations

## Current limitation: 
greedy runs for both parties produce symmetric seat counts under the current objective structure. Further objective differentiation and constraint tuning are required.

## Deployment Model

This project is designed for static deployment.

All run outputs must be written inside:

apps/web/public/outputs/

This allows:

Local preview

Easy Vercel deployment

No backend server

Configuration

Users must provide:

Precinct election shapefile path

Unit ID column name

CRS

Algorithm parameters

A config.example.yaml file is provided as a template. Do not commit config.yaml if it contains absolute paths.

Work In Progress and TODO
Algorithm Improvements

Fix greedy objective symmetry

Improve maximize logic sensitivity

Fix contiguity constraint enforcement

Get constrained KMeans running

Refactor population balancing logic

Add objective score tracking

Add compactness metrics

Frontend Improvements

Add mouseover capability back

Improve tooltip styling

Add margin intensity shading

Add compactness metrics display

Add seat count summary banner

Add algorithm comparison view

Architecture Improvements

Convert to Pack v2 format

Add run metadata JSON

Add reproducibility seed tracking

Add performance profiling

### Research Intent

This project is intended as:

A computational redistricting experimentation platform

A portfolio demonstration of geospatial data processing and optimization

A sandbox for testing partisan objectives and constraints

It is not intended as a production-grade redistricting system.

## Disclaimer

Redistricting is a politically and legally complex process. This project is a computational experiment and research tool.

All data remains subject to the licensing terms of the Redistricting Data Hub.


Built as part of an independent computational research and portfolio project focused on algorithmic redistricting and political data systems.