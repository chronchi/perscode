perscode
===

Vectorization methods for persistence diagrams based in the paper [Persistence Codebooks for
Topological Data Analysis](https://arxiv.org/abs/1802.04852).

## Installation

```python
pip install perscode
```

## Usage

```python
import perscode
import numpy as np

# generate diagrams
diagrams = [np.random.rand(100,2) for _ in range(20)]
for diagram in diagrams:
    diagram[:,1] += diagram[:,0]

# N is the size of the vectors
# normalize is a Bool to whether or not normalize the output vector
pbow = perscode.PBoW(N = 3, normalize = False)
wpbow = perscode.wPBoW(N = 3)
# n_subsample is an int or None. If none all points will be used when calculating GMMs.
spbow = perscode.sPBoW(N = 10, n_subsample = None)

# vectorize diagrams
pbow_diagrams  = pbow.transform(diagrams)
wpbow_diagrams = wpbow.transform(diagrams)
spbow_diagrams = spbow.transform(diagrams)

# for PVLAD and stable PVLAD
pvlad = perscode.PVLAD(N = 3)
spvlad = perscode.sPVLAD(N = 3)

pvlad_diagrams = pvlad.transform(diagrams)
spvlad_diagrams = spvlad.transform(diagrams)
```

## TODO
- [x] Implement options to pass cluster centers as arguments in wPBoW and sPBoW.
- [x] Implement PVLAD
- [x] Implement sPVLAD
- [ ] Implement PFV
- [x] Implement optional weighted subsampling to wPBoW, sPBoW, sPVLAD classes.
- [ ] Proper documentation
