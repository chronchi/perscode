perscode
===

Vectorization methods for persistence diagrams based in the paper [Persistence Codebooks for
Topological Data Analysis](https://arxiv.org/abs/1802.04852).

## Usage

```python
import perscode
import numpy as np

# N is the size of the vectors
# normalize is a Bool to whether or not normalize the output vector
pbow = perscode.PBoW(N = 50, normalize = False)

# generate diagrams
diagrams = [np.random.rand(100,2) for _ in range(20)]
for diagram in diagrams:
    diagram[:,1] += diagram[:,0]

# vectorize diagrams
pbow_diagrams = pbow.transform(diagrams)
```

## TODO
- [ ] Implement options to pass cluster centers as arguments in wPBoW and sPBoW.
- [ ] Implement PVLAD
- [ ] Implement sPVLAD
- [ ] Implement PFV
 
