# !!WORK IN PROGESS!!
# Preprocessing of Whole Slides Images (WSI) histology data
Preprocess whole slides images and extracting tiles:
- Extracting tiles of H&E histology slides at different magnifications

# Project organization

    ├── README.md
    ├── config
    │   └── config_test.yaml
    │
    ├── data
    │   ├── tiles       
    │   ├── processed      
    │   └── raw  
    │
    ├── images 
    │
    ├── requirements.txt
    │
    ├── setup.py           
    │
    ├── src
    │   ├── __init__.py
    │   │
    │   ├── utils.py
    │   │
    │   └── visualization
    │       └── visualize.py
    │
    └── ...

### Dataset:
Can found test examples on [TCGA portal](https://portal.gdc.cancer.gov/repository)

## Table of contents

## Getting Started

## Prerequisites
- [Python](https://www.python.org/downloads/) >= 3.7.6 (tested)
- [OpenSlide](https://github.com/openslide/openslide-python) [1]
- [QuPath](https://qupath.github.io/) [2]

## Installation

## Documentation

## Example
![Alt text](./images/histo_process_pipeline.png?raw=true "Preprocessing pipeline")

## Results

## References
- [1] Goode, A., Gilbert, B., Harkes, J., Jukic, D., & Satyanarayanan, M. (2013). OpenSlide: A vendor-neutral software foundation for digital pathology. Journal of pathology informatics, 4. https://doi.org/10.4103/2153-3539.119005
- [2] Bankhead, P. et al. QuPath: Open source software for digital pathology image analysis. Scientific Reports (2017). https://doi.org/10.1038/s41598-017-17204-5

