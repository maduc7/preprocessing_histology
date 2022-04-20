# !!WORK IN PROGESS!!
# Preprocessing of Whole Slides Images (WSI) histology data
Preprocess whole slides images and extracting tiles:
- Extracting tiles of H&E histology slides at different magnifications

# Project organization

    ├── README.md          
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


## The Team
- Marie Duc: PhD Student at the Werner Siemens Imaging Center in Tübingen

## License and citing

_License_: Check the [License](https://github.com/maduc7/preprocessing_histology/LICENSE) (MIT)

_Citing_: 

    @misc{preprocessing_histo_2022,
        authors       = {Marie Duc},
        title        = {Preprocessing of WSI},
        year         = {2022},
        version      = {1.0},
        publisher    = {GitHub},
        journal      = {GitHub repositiory},
        howpublished = {\url{https://github.com/maduc7/preprocessing_histology}},
        commit       = {XXX}
        }
