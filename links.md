# Paper Links and Abstracts

## 2023
### [Equivariant Flow Matching with Hybrid Probability Transport for 3D Molecule Generation](https://arxiv.org/pdf/2312.07168)

#### Abstract
The generation of 3D molecules requires simultaneously deciding the categorical features (atom types) and continuous features (atom coordinates). Deep generative models, especially Diffusion Models (DMs), have demonstrated effectiveness in generating feature-rich geometries. However, existing DMs typically suffer from
unstable probability dynamics with inefficient sampling speed. In this paper, we introduce geometric flow matching, which enjoys the advantages of both equivariant modeling and stabilized probability dynamics. More specifically, we propose a hybrid probability path where the coordinates probability path is regularized by an equivariant optimal transport, and the information between different modalities is aligned. Experimentally, the proposed method could consistently achieve better performance on multiple molecule generation benchmarks with 4.75× speed up of sampling on average.

### [Multimarginal Generative Modeling with Stochastic Interpolants](https://arxiv.org/pdf/2310.03695)

#### Abstract
Given a set of $K$ probability densities, we consider the multimarginal generative modeling problem of learning a joint distribution that recovers these densities as marginals. The structure of this joint distribution should identify multi-way correspondences among the prescribed marginals. We formalize an approach to this task within a generalization of the stochastic interpolant framework, leading to efficient learning algorithms built upon dynamical transport of measure. Our generative models are defined by velocity and score fields that can be characterized as the minimizers of simple quadratic objectives, and they are defined on a simplex that generalizes the time variable in the usual dynamical transport framework. The resulting transport on the simplex is influenced by all marginals, and we show that multi-way correspondences can be extracted. The identification of such correspondences has applications to style transfer, algorithmic fairness, and data decorruption. In addition, the multimarginal perspective enables an efficient algorithm for reducing the dynamical transport cost in the ordinary two-marginal setting. We demonstrate these capacities with several numerical examples.

## 2024
### [Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/pdf/2402.04997)

#### Abstract
Combining discrete and continuous data is an important capability for generative models. We
present Discrete Flow Models (DFMs), a new flow-based model of discrete data that provides
the missing link in enabling flow-based gener ative models to be applied to multimodal con-
tinuous and discrete data problems. Our key insight is that the discrete equivalent of continuous
space flow matching can be realized using Continuous Time Markov Chains. DFMs benefit from a
simple derivation that includes discrete diffusion models as a specific instance while allowing im-
proved performance over existing diffusion-based approaches. We utilize our DFMs method to build
a multimodal flow-based modeling framework. We apply this capability to the task of protein
co-design, wherein we learn a model for jointly generating protein structure and sequence. Our
approach achieves state-of-the-art co-design performance while allowing the same multimodal
model to be used for flexible generation of the sequence or structure.

### [SemlaFlow - Efficient 3D Molecular Generation with Latent Attention and Equivariant Flow Matching](https://arxiv.org/pdf/2406.07266)

#### Abstract
Methods for jointly generating molecular graphs along with their 3D conformations have gained prominence recently due to their potential impact on structure-based drug design. Current approaches, however, often suffer from very slow sampling times or generate molecules with poor chemical validity. Addressing these limitations, we propose Semla, a scalable E(3)-equivariant message passing architecture. We further introduce an unconditional 3D molecular generation model, SemlaFlow, which is trained using equivariant flow matching to generate a joint distribution over atom types, coordinates, bond types and formal charges. Our model produces state-of-the-art results on benchmark datasets with as few as 20 sampling steps, corresponding to a two order-of-magnitude speedup compared to state-of-the-art. Furthermore, we highlight limitations of current evaluation methods for 3D generation and propose new benchmark metrics for unconditional molecular generators. Finally, using these new metrics, we compare our model’s ability to generate high quality samples against current approaches and further demonstrate SemlaFlow’s strong performance.

### [Stochastic Interpolants with Data-Dependent Couplings](https://arxiv.org/pdf/2310.03725)

#### Abstract
Generative models inspired by dynamical transport of measure – such as flows and diffusions – construct a continuous-time map between two probability densities. Conventionally, one of these is the target density, only accessible through samples, while the other is taken as a simple base density that is data-agnostic. In this work, using the framework of stochastic interpolants, we formalize how to couple the base and the target densities, whereby samples from the base are computed conditionally given samples from the target in a way that is different from (but does not preclude) incorporating information about class labels or continuous embeddings. This enables us to construct dynamical transport maps that serve as conditional generative models. We show that these transport maps can be learned by solving a simple square loss regression problem analogous to the standard independent setting. We demonstrate the usefulness of constructing dependent couplings in practice through experiments in super-resolution and in-painting. The code is available at [https://github.com/interpolants/couplings](https://github.com/interpolants/couplings).

### [Flow Matching Guide and Code](https://arxiv.org/pdf/2412.06264)

#### Abstract
Flow Matching (FM) is a recent framework for generative modeling that has achieved state-of-the-art performance across various domains, including image, video, audio, speech, and biological structures. This guide offers a comprehensive and self-contained review of FM, covering its mathematical foundations, design choices, and extensions. By also providing a PyTorch package featuring relevant examples (e.g., image and text generation), this work aims to serve as a resource for both novice and experienced researchers interested in understanding, applying and further developing FM.

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