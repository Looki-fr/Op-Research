# Op-Research

## 1. Introduction

This repository contains the code for the project of the course Operations Research (2023-2024) at Efrei Paris. The goal of this project is to solve a problem of transportation using different methods and algorithms.

## 2. How to use

### 2.1. Requirements

To run the code, you need to have Python 3 installed on your computer. You can download it [here](https://www.python.org/downloads/).

### 2.2. Installation

First, clone the repository on your computer:

```bash
git clone
```

Then, install the required packages:

```bash
pip install -r requirements.txt
```

### 2.3. Usage

You can run the code by executing the following command:

```bash
python main.py
```

### 2.4 Data format

The data is stored in the `data` folder. The data is stored in a TXT file with the following format:

```
m n
c11 c12 ... c1n d1
c21 c22 ... c2n d2
...
cm1 cm2 ... cmn dm
s1 s2 ... sn
```

Where:
```
- m is the number of suppliers
- n is the number of clients
- cij is the cost of transporting one unit from supplier i to client j
- di is the demand of client i
- si is the supply of supplier i
```

## 3. Authors

- Louis Le Meilleur
- Paul Mairesse
- Axel Loones
- Joseph Bénard

