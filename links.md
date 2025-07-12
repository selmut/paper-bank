# Papers

## 2025
### [TABASCO: A Fast, Simplified Model for Molecular Generation with Improved Physical Quality](https://arxiv.org/pdf/2507.00899)

#### Abstract
State-of-the-art models for 3D molecular generation are based on significant inductive biases—SE(3), permutation equivariance to respect symmetry and graph message -passing networks to capture local chemistry—yet the generated molecules still struggle with physical plausibility. We introduce TABASCO which relaxes these assumptions: The model has a standard non-equivariant transformer architecture, treats atoms in a molecule as sequences and reconstructs bonds deterministically after generation. The absence of equivariant layers and message passing allows us to significantly simplify the model architecture and scale data throughput. On the GEOM - Drugs benchmark TABASCO achieves state-of-the-art PoseBusters validity and delivers inference roughly 10× faster than the
strongest baseline, while exhibiting emergent rotational equivariance despite symmetry not being hard - coded. Our work offers a blueprint for training minimalist, high-throughput generative models suited to specialised tasks such as structure - and pharmacophore - based drug design. We provide a link to our implementation at [ github.com/carlosinator/tabasco]( github.com/carlosinator/tabasco).

### [Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule](https://arxiv.org/pdf/2505.07286)