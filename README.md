# Pharmacophore-Guided Scaffold Hopping

A workflow for identifying and generating structurally diverse analogs while preserving key molecular interactions using pharmacophore matching.

## Overview

This project implements a pharmacophore-guided scaffold hopping algorithm that:
1. Extracts pharmacophore features from reference and scaffold molecules
2. Identifies feature correspondences between molecules
3. Aligns the scaffold to the reference pharmacophore
4. Generates new molecules via scaffold replacement

The generated molecules can also be used as inputs to the FARE GitHub repository to identify suitable fragment replacements, if desired: [FARE](https://github.com/CathereneTomy/FARE/tree/main)

## Installation

### Option 1: Using Conda (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/CathereneTomy/Scaffold_Hop.git
   cd Scaffold_Hop
   ```

2. Create the environment from the YAML file:
   ```bash
   conda env create -f environment_scaffold_hop.yml
   ```

3. Activate the environment:
   ```bash
   conda activate scaffold_hop
   ```

### Option 2: Using pip

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create a Python 3.14 virtual environment (recommended: use conda)

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

### Using Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `scaffold_hop.ipynb` in your browser

3. Run the cells sequentially to execute the scaffold hopping workflow

### Example Usage

The notebook demonstrates scaffold hopping with:

- **Reference molecule**: `Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1`
- **Scaffold**: `C1=CNC2=CN=CN=C21`

Key functions:
- `ph_similarity_pipeline()` - Computes pharmacophore similarity
- `align_and_visualize()` - Aligns reference and scaffold
- `build_hopped_scaffold()` - Generates scaffold-hopped molecules

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies (pip)
├── environment_scaffold_hop.yml # Conda environment definition
├── scaffold_hop.ipynb          # Main Jupyter notebook
├── utils.py                    # Core utility functions
└── alignment.png              # Example output
```

## Output

The pipeline produces:
- Pharmacophore alignment visualization
- Scaffold-hopped molecule SMILES
- RMSD and query coverage metrics
- 2D molecular visualizations
