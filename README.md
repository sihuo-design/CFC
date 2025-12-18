# CFC

**The official code for "Coarse-to-Fine Open-Set Graph Node Classification with Large Language Models".**



## Overview

CFC (Coarse-to-Fine Classification) is an open-set classification framework. It enables:

- **In-distribution (ID) classification**  
- **Out-of-distribution (OOD) detection and classification**  

Unlike existing methods that treat all OOD samples as a single class, CFC can provide **probable OOD labels** for improved interpretability and practical utility in applications.



## Installation

1. **Clone the repository**

```bash
git clone https://github.com/sihuo-design/CFC.git
cd CFC
```

2. **Create and activate a Conda environment**

```bash
conda create -n cfc_env python=3.8
conda activate cfc_env

```
```bash
pip install -r requirements.txt
```

## Usage

**Run experiments on graph datasets**

```bash
python run.sh --dataset Cora
```