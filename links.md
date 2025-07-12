# Papers

## 2025
### [TABASCO: A Fast, Simplified Model for Molecular Generation with Improved Physical Quality](https://arxiv.org/pdf/2507.00899)

#### Abstract
State-of-the-art models for 3D molecular generation are based on significant inductive biases—SE(3), permutation equivariance to respect symmetry and graph message -passing networks to capture local chemistry—yet the generated molecules still struggle with physical plausibility. We introduce TABASCO which relaxes these assumptions: The model has a standard non-equivariant transformer architecture, treats atoms in a molecule as sequences and reconstructs bonds deterministically after generation. The absence of equivariant layers and message passing allows us to significantly simplify the model architecture and scale data throughput. On the GEOM - Drugs benchmark TABASCO achieves state-of-the-art PoseBusters validity and delivers inference roughly 10× faster than the
strongest baseline, while exhibiting emergent rotational equivariance despite symmetry not being hard - coded. Our work offers a blueprint for training minimalist, high-throughput generative models suited to specialised tasks such as structure - and pharmacophore - based drug design. We provide a link to our implementation at [ github.com/carlosinator/tabasco]( github.com/carlosinator/tabasco).

### [Piloting Structure-Based Drug Design via Modality-Specific Optimal Schedule](https://arxiv.org/pdf/2505.07286)

#### Abstract
Structure-Based Drug Design (SBDD) is crucial for identifying bioactive molecules. Recent deep generative models are faced with challenges in geometric structure modeling. A major bottle neck lies in the twisted probability path of multimodalities—continuous 3D positions and discrete 2D topologies—which jointly determine molecular geometries. By establishing the fact that noise schedules decide the Variational Lower Bound (VLB) for the twisted probability path, we propose VLB-Optimal Scheduling (VOS) strategy in
this under-explored area, which optimizes VLB as a path integral for SBDD. Our model effectively enhances molecular geometries and interaction modeling, achieving a state-of-the-art PoseBusters passing rate of 95.9% on CrossDock, more than 10% improvement upon strong baselines, while maintaining high affinities and robust
intramolecular validity evaluated on a held-out test set. Code is available at [https://github.com/AlgoMole/MolCRAFT](https://github.com/AlgoMole/MolCRAFT).


### [Applications of Modular Co-Design for De Novo 3D Molecule Generation](https://arxiv.org/pdf/2505.18392)

#### Abstract
De novo 3D molecule generation is a pivotal task in drug discovery. However, many recent geometric generative models struggle to produce high-quality 3D structures, even if they maintain 2D validity and topological stability. To tackle this issue and enhance the learning of effective molecular generation dynamics, we present Megalodon–a family of scalable transformer models. These models are enhanced with basic equivariant layers and trained using a joint continuous and discrete denoising co-design objective. We assess Megalodon’s performance on established molecule generation benchmarks and introduce new 3D structure benchmarks that evaluate a model’s capability to generate realistic molecular structures, particularly focusing on energetics. We show that Megalodon achieves state-of-the-art results in 3D molecule generation, conditional structure generation, and structure energy benchmarks using diffusion and flow matching. Furthermore, doubling the number of parameters in Megalodon to 40M significantly enhances its performance, generating up to 49x more valid large molecules and achieving energy levels that are 2-10x lower than those of the best prior generative models.