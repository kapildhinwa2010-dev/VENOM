# ARCHIVE — the sibling repositories

The master plan's TABLE 2 ended with *"baaki ~106 repos: archive, README mein link rakho"*.
This file is that index. It is generated from the GitHub API, so it can be
refreshed at any time; the list below is a snapshot.

**20 of these are consumed by VENOM** — not re-implemented, called through the
optional adapters in `venom/integrations/`. See
[`venom/integrations/README.md`](venom/integrations/README.md) for what each one
provides and what it does NOT prove.

## Consumed by VENOM (20)

| Repo | What it provides | Description |
|---|---|---|
| [`BUS-TRACE`](https://github.com/kapildhinwa2010-dev/BUS-TRACE) | see integrations README | Analyze residual-stream writes block by block: report per-block L2 write norms and shares, |
| [`ComplexityRouter`](https://github.com/kapildhinwa2010-dev/ComplexityRouter) | see integrations README | Route by complexity |
| [`DEAD-CELL`](https://github.com/kapildhinwa2010-dev/DEAD-CELL) | see integrations README | Python tool for measuring exact-zero rows, columns, and parameters in weight matrices, plu |
| [`Draft_ponit_harness`](https://github.com/kapildhinwa2010-dev/Draft_ponit_harness) | see integrations README | A PyTorch harness that measures the first generated token where a low-rank delta replay di |
| [`FactBase`](https://github.com/kapildhinwa2010-dev/FactBase) | see integrations README | Fact storage with zero overhead |
| [`fluidtech`](https://github.com/kapildhinwa2010-dev/fluidtech) | see integrations README | A bounded memory layer for language models, with exact recall for retained token IDs. NumP |
| [`GEOMETRY-SCAN`](https://github.com/kapildhinwa2010-dev/GEOMETRY-SCAN) | see integrations README | A Python toolkit for measuring per-layer model hidden-state geometry: covariance-based int |
| [`INNER-VOTE-`](https://github.com/kapildhinwa2010-dev/INNER-VOTE-) | see integrations README | Python package for measuring hidden-state consensus across paraphrases, reporting layer-wi |
| [`KNOW-VAULT`](https://github.com/kapildhinwa2010-dev/KNOW-VAULT) | see integrations README | Estimate geometric memory-slot capacity, effective rank, dead neurons, and activation occu |
| [`LAYER-CENSUS-`](https://github.com/kapildhinwa2010-dev/LAYER-CENSUS-) | see integrations README | Python tool for producing per-layer weight-matrix atlases with shape, parameter count, Fro |
| [`LOGIT-STREAM`](https://github.com/kapildhinwa2010-dev/LOGIT-STREAM) | see integrations README | LOGIT-STREAM analyzes layer-prefix logits from a shared readout, decomposing a token’s fin |
| [`NEVER_LIE`](https://github.com/kapildhinwa2010-dev/NEVER_LIE) | see integrations README | Python memory library that verifies stored facts with SHA-256 and abstains when data corru |
| [`NOISE-STORM`](https://github.com/kapildhinwa2010-dev/NOISE-STORM) | see integrations README | A Python tool for mapping block-level fragility: inject seeded Gaussian noise scaled to ea |
| [`POWERLAW-PROBE`](https://github.com/kapildhinwa2010-dev/POWERLAW-PROBE) | see integrations README | CLI tool that compares power-law exponents of a language model layer’s activation and weig |
| [`SKIP-RUN-`](https://github.com/kapildhinwa2010-dev/SKIP-RUN-) | see integrations README | A Python tool for testing effective depth in additive-block stacks by skipping every k-th  |
| [`STATE-MIRROR`](https://github.com/kapildhinwa2010-dev/STATE-MIRROR) | see integrations README | Compare layer-wise hidden states from aligned twin prompts, reporting mean token cosine si |
| [`TMU`](https://github.com/kapildhinwa2010-dev/TMU) | see integrations README | Experimental GPT-2 plugin layer for modeling external world time and internal AI compute t |
| [`TRACE-PATH`](https://github.com/kapildhinwa2010-dev/TRACE-PATH) | see integrations README | A Python activation-patching tool that traces information flow through stream-additive blo |
| [`TWIN-TOWER`](https://github.com/kapildhinwa2010-dev/TWIN-TOWER) | see integrations README | Python tool that detects near-duplicate same-shape neural-network weight layers using abso |
| [`VARIEANCE-LIES`](https://github.com/kapildhinwa2010-dev/VARIEANCE-LIES) | see integrations README | A PyTorch probe comparing low-rank activation-delta variance capture with next-token agree |

## The rest (111)

Archived: not imported into VENOM. They remain the source of truth for
their own topic.

| Repo | Language | Description |
|---|---|---|
| [`-KNOCKOUT-LAB`](https://github.com/kapildhinwa2010-dev/-KNOCKOUT-LAB) | Python | A Python ablation lab for stream-additive models. Removes each block in turn, measures sco |
| [`Activation-HELIX`](https://github.com/kapildhinwa2010-dev/Activation-HELIX) | Python | Activation analysis + HELIX storage |
| [`Activation-Lib`](https://github.com/kapildhinwa2010-dev/Activation-Lib) | Python | Activation library |
| [`Activation-Research`](https://github.com/kapildhinwa2010-dev/Activation-Research) | Python | Activation research |
| [`ActivationCompress`](https://github.com/kapildhinwa2010-dev/ActivationCompress) | Python | Compress neural activations |
| [`ActivationRank`](https://github.com/kapildhinwa2010-dev/ActivationRank) | Python | Rank activations by importance |
| [`AliasDeriver`](https://github.com/kapildhinwa2010-dev/AliasDeriver) | Python | Derive aliases from facts |
| [`ANT-HILL`](https://github.com/kapildhinwa2010-dev/ANT-HILL) | Python | Python tool that maps language-model activation flow across layers and token positions int |
| [`Behavior-HELIX`](https://github.com/kapildhinwa2010-dev/Behavior-HELIX) | Python | Behavior testing + knowledge derivation |
| [`Behavior-Lib`](https://github.com/kapildhinwa2010-dev/Behavior-Lib) | Python | Behavior library |
| [`Behavior-Research`](https://github.com/kapildhinwa2010-dev/Behavior-Research) | Python | Behavior research |
| [`BehaviorDNA`](https://github.com/kapildhinwa2010-dev/BehaviorDNA) | Python | Extract behavior patterns from activations |
| [`BottleneckTest`](https://github.com/kapildhinwa2010-dev/BottleneckTest) | Python | Test model bottlenecks |
| [`CausalMemory`](https://github.com/kapildhinwa2010-dev/CausalMemory) | Python | Causal reasoning memory |
| [`ChainReasoner`](https://github.com/kapildhinwa2010-dev/ChainReasoner) | Python | Chain reasoning engine |
| [`ChunkGen`](https://github.com/kapildhinwa2010-dev/ChunkGen) | Python | Chunked matrix generation |
| [`Compression-HELIX`](https://github.com/kapildhinwa2010-dev/Compression-HELIX) | Python | Compression + knowledge memory |
| [`Compression-Lib`](https://github.com/kapildhinwa2010-dev/Compression-Lib) | Python | Compression library |
| [`Compression-Research`](https://github.com/kapildhinwa2010-dev/Compression-Research) | Python | Compression research |
| [`Compression-Serve`](https://github.com/kapildhinwa2010-dev/Compression-Serve) | Python | Compression as a service |
| [`CompressionBench`](https://github.com/kapildhinwa2010-dev/CompressionBench) | Python | Compression benchmark suite |
| [`CompressionTest`](https://github.com/kapildhinwa2010-dev/CompressionTest) | Python | Test compression methods |
| [`CompressorNet`](https://github.com/kapildhinwa2010-dev/CompressorNet) | Python | Neural network compression benchmark suite |
| [`ContentAddress`](https://github.com/kapildhinwa2010-dev/ContentAddress) | Python | Content-addressed storage |
| [`DenseTap`](https://github.com/kapildhinwa2010-dev/DenseTap) | Python | Every transformer layer reads every previous layer — for 20 extra scalars. NumPy reference |
| [`DistributionShift`](https://github.com/kapildhinwa2010-dev/DistributionShift) | Python | Detect distribution shifts |
| [`DNA-CNN`](https://github.com/kapildhinwa2010-dev/DNA-CNN) | Python | CNN with DNA-generated filters |
| [`DNA-Encoder`](https://github.com/kapildhinwa2010-dev/DNA-Encoder) | Python | Experimental PyTorch tooling for Fourier-feature parameter encoding and weight reconstruct |
| [`DNA-GAN`](https://github.com/kapildhinwa2010-dev/DNA-GAN) | Python | GAN with compressed generator |
| [`DNA-HELIX`](https://github.com/kapildhinwa2010-dev/DNA-HELIX) | Python | DNA compression + HELIX memory |
| [`DNA-Lib`](https://github.com/kapildhinwa2010-dev/DNA-Lib) | Python | DNA library |
| [`DNA-Lite`](https://github.com/kapildhinwa2010-dev/DNA-Lite) | Python | Lightweight DNA for small models |
| [`DNA-LLM`](https://github.com/kapildhinwa2010-dev/DNA-LLM) | Python | A failed project |
| [`DNA-LLM-PART-2`](https://github.com/kapildhinwa2010-dev/DNA-LLM-PART-2) | Python | — |
| [`DNA-LLM-PART-3`](https://github.com/kapildhinwa2010-dev/DNA-LLM-PART-3) | Python | A failed project |
| [`DNA-MLP`](https://github.com/kapildhinwa2010-dev/DNA-MLP) | Python | MLP with DNA compression |
| [`DNA-Pro`](https://github.com/kapildhinwa2010-dev/DNA-Pro) | Python | Production-ready DNA generator |
| [`DNA-Research`](https://github.com/kapildhinwa2010-dev/DNA-Research) | Python | DNA research framework |
| [`DNA-RNN`](https://github.com/kapildhinwa2010-dev/DNA-RNN) | Python | RNN with DNA compression |
| [`DNA-Serve`](https://github.com/kapildhinwa2010-dev/DNA-Serve) | Python | DNA generation as a service |
| [`DNA-Transformer`](https://github.com/kapildhinwa2010-dev/DNA-Transformer) | Python | Transformer with DNA-generated weights |
| [`DNA_DYNIMO`](https://github.com/kapildhinwa2010-dev/DNA_DYNIMO) | Python | Research toolkit exploring Fourier-feature neural fields for matrix generation, budget-awa |
| [`ECC-Memory`](https://github.com/kapildhinwa2010-dev/ECC-Memory) | Python | Error-correcting code memory |
| [`fluid-ARC`](https://github.com/kapildhinwa2010-dev/fluid-ARC) | Python | Research/prototype Python package exploring a transformer-attention alternative: cellular- |
| [`Fourier-HELIX`](https://github.com/kapildhinwa2010-dev/Fourier-HELIX) | Python | Fourier features + spiral addressing |
| [`FourierNet`](https://github.com/kapildhinwa2010-dev/FourierNet) | Python | Experimental PyTorch Fourier-feature generator for coordinate-based weight matrices. |
| [`FREQUENCY_EVER`](https://github.com/kapildhinwa2010-dev/FREQUENCY_EVER) | Python | Experimental wave-native AI architecture using tokenizer-free frequency-domain computation |
| [`Generalization-Research`](https://github.com/kapildhinwa2010-dev/Generalization-Research) | Python | Generalization research |
| [`GeneralizationTest`](https://github.com/kapildhinwa2010-dev/GeneralizationTest) | Python | Test model generalization |
| [`GradientFlow`](https://github.com/kapildhinwa2010-dev/GradientFlow) | Python | Track gradient flow through layers |
| [`HEAD-DETECTIVE`](https://github.com/kapildhinwa2010-dev/HEAD-DETECTIVE) | Python | A pure-NumPy Python tool that classifies attention heads from attention matrices using pat |
| [`HELIX-Core`](https://github.com/kapildhinwa2010-dev/HELIX-Core) | Python | Core HELIX implementation |
| [`HELIX-Lib`](https://github.com/kapildhinwa2010-dev/HELIX-Lib) | Python | HELIX library |
| [`HELIX-Pro`](https://github.com/kapildhinwa2010-dev/HELIX-Pro) | Python | Production HELIX |
| [`HELIX-Research`](https://github.com/kapildhinwa2010-dev/HELIX-Research) | Python | HELIX research framework |
| [`HELIX-Serve`](https://github.com/kapildhinwa2010-dev/HELIX-Serve) | Python | HELIX memory as a service |
| [`HierarchicalMemory`](https://github.com/kapildhinwa2010-dev/HierarchicalMemory) | Python | Hierarchical memory system |
| [`HybridRouter`](https://github.com/kapildhinwa2010-dev/HybridRouter) | Python | Hybrid routing |
| [`InfiniteGen`](https://github.com/kapildhinwa2010-dev/InfiniteGen) | Python | Infinite parameter generator |
| [`InterferenceEngine`](https://github.com/kapildhinwa2010-dev/InterferenceEngine) | Python | Interference-based retrieval |
| [`Kapildhinwa2010-dev`](https://github.com/kapildhinwa2010-dev/Kapildhinwa2010-dev) | — | Kapil Dhinwa’s AI and machine learning projects, experiments, and open-source research. |
| [`Knowledge-Lib`](https://github.com/kapildhinwa2010-dev/Knowledge-Lib) | Python | Knowledge library |
| [`Knowledge-Research`](https://github.com/kapildhinwa2010-dev/Knowledge-Research) | Python | Knowledge research |
| [`Knowledge-Serve`](https://github.com/kapildhinwa2010-dev/Knowledge-Serve) | Python | Knowledge as a service |
| [`KnowledgeBench`](https://github.com/kapildhinwa2010-dev/KnowledgeBench) | Python | Knowledge benchmark suite |
| [`KnowledgeGraph`](https://github.com/kapildhinwa2010-dev/KnowledgeGraph) | Python | Graph-based knowledge with HELIX |
| [`KnowledgeProbe`](https://github.com/kapildhinwa2010-dev/KnowledgeProbe) | Python | Probe model knowledge in activations |
| [`KnowledgeRouter`](https://github.com/kapildhinwa2010-dev/KnowledgeRouter) | Python | Route by knowledge |
| [`LLM-EQUATION`](https://github.com/kapildhinwa2010-dev/LLM-EQUATION) | Python | Fits measured AI curves to closed-form equations, selects models by BIC, and tests predict |
| [`MatrixGen`](https://github.com/kapildhinwa2010-dev/MatrixGen) | Python | Coordinate-based PyTorch neural field for memory-aware matrix generation. |
| [`Memory-Lib`](https://github.com/kapildhinwa2010-dev/Memory-Lib) | Python | Memory library |
| [`Memory-Research`](https://github.com/kapildhinwa2010-dev/Memory-Research) | Python | Memory research |
| [`Memory-Serve`](https://github.com/kapildhinwa2010-dev/Memory-Serve) | Python | Service |
| [`MemoryBench`](https://github.com/kapildhinwa2010-dev/MemoryBench) | Python | Memory benchmark suite |
| [`MemoryRouter`](https://github.com/kapildhinwa2010-dev/MemoryRouter) | Python | Route by memory |
| [`MemoryTest`](https://github.com/kapildhinwa2010-dev/MemoryTest) | Python | Test memory usage |
| [`MOLECULE-COMPRESS`](https://github.com/kapildhinwa2010-dev/MOLECULE-COMPRESS) | Python | Measures quality and size effects of post-training ternary quantization on Qwen2.5-0.5B, w |
| [`NEXUS-16`](https://github.com/kapildhinwa2010-dev/NEXUS-16) | Python | Wave-native, tokenizer-free byte-level semantic encoder using frequency representations an |
| [`OutputSensitivity`](https://github.com/kapildhinwa2010-dev/OutputSensitivity) | Python | From b4.py |
| [`ParallelGen`](https://github.com/kapildhinwa2010-dev/ParallelGen) | Python | Parallel generation |
| [`ParamShrink`](https://github.com/kapildhinwa2010-dev/ParamShrink) | Python | Experimental PyTorch parameter reconstruction using coordinate and Fourier features. |
| [`PROBE-STACK`](https://github.com/kapildhinwa2010-dev/PROBE-STACK) | Python | A Python tool for training seeded logistic probes on model hidden states and comparing per |
| [`QuantDNA`](https://github.com/kapildhinwa2010-dev/QuantDNA) | Python | Quantized DNA weight generation |
| [`Robustness-Lib`](https://github.com/kapildhinwa2010-dev/Robustness-Lib) | Python | Robustness library |
| [`Robustness-Research`](https://github.com/kapildhinwa2010-dev/Robustness-Research) | Python | Robustness research |
| [`RobustnessBench`](https://github.com/kapildhinwa2010-dev/RobustnessBench) | Python | Robustness benchmark suite |
| [`RobustnessTest`](https://github.com/kapildhinwa2010-dev/RobustnessTest) | Python | Test model robustness |
| [`Router-Lib`](https://github.com/kapildhinwa2010-dev/Router-Lib) | Python | Router library |
| [`Router-Serve`](https://github.com/kapildhinwa2010-dev/Router-Serve) | Python | Routing as a service |
| [`RULE-FORAGER`](https://github.com/kapildhinwa2010-dev/RULE-FORAGER) | Python | A pure-Python tool that mines inverse-relation and shared-filler patterns from factual tri |
| [`RuleEngine`](https://github.com/kapildhinwa2010-dev/RuleEngine) | Python | Rule-based knowledge derivation |
| [`SafeGen`](https://github.com/kapildhinwa2010-dev/SafeGen) | Python | Memory-safe generation |
| [`SemanticMemory`](https://github.com/kapildhinwa2010-dev/SemanticMemory) | Python | Semantic memory system |
| [`SensitivityMap`](https://github.com/kapildhinwa2010-dev/SensitivityMap) | Python | Map output sensitivity |
| [`SparseDNA`](https://github.com/kapildhinwa2010-dev/SparseDNA) | Python | Sparse matrix generation with DNA |
| [`Speed-Lib`](https://github.com/kapildhinwa2010-dev/Speed-Lib) | Python | Speed library |
| [`Speed-Research`](https://github.com/kapildhinwa2010-dev/Speed-Research) | Python | Speed research |
| [`Speed-Serve`](https://github.com/kapildhinwa2010-dev/Speed-Serve) | Python | Service |
| [`SpeedBench`](https://github.com/kapildhinwa2010-dev/SpeedBench) | Python | Speed benchmark suite |
| [`SpeedRouter`](https://github.com/kapildhinwa2010-dev/SpeedRouter) | Python | Route by speed |
| [`SpeedTest`](https://github.com/kapildhinwa2010-dev/SpeedTest) | Python | Test inference speed |
| [`Spiking_GNN`](https://github.com/kapildhinwa2010-dev/Spiking_GNN) | Python | Bio-inspired spiking graph neural network with neuromorphic learning and connectome-inspir |
| [`SpiralDB`](https://github.com/kapildhinwa2010-dev/SpiralDB) | Python | Spiral-addressed database |
| [`StreamGen`](https://github.com/kapildhinwa2010-dev/StreamGen) | Python | Streaming generation |
| [`SVD-Lens`](https://github.com/kapildhinwa2010-dev/SVD-Lens) | Python | SVD-based activation analysis |
| [`TemporalMemory`](https://github.com/kapildhinwa2010-dev/TemporalMemory) | Python | Temporal knowledge storage |
| [`VarianceBasis`](https://github.com/kapildhinwa2010-dev/VarianceBasis) | Python | Variance-based feature extraction |
| [`VENOM`](https://github.com/kapildhinwa2010-dev/VENOM) | Python | THE PARASITE |
| [`WAVE-CLASH`](https://github.com/kapildhinwa2010-dev/WAVE-CLASH) | Python | Python tool for evaluating bit superposition on orthogonal DFT carriers with crosstalk and |
| [`WaveMemory`](https://github.com/kapildhinwa2010-dev/WaveMemory) | Python | Wave lattice memory system |
| [`WeightForge`](https://github.com/kapildhinwa2010-dev/WeightForge) | Python | Experimental PyTorch utility for coordinate-based neural-weight generation. |

---

_Snapshot: 131 repositories on the account, 20 consumed by VENOM, 111 archived._

Regenerate with:

```bash
python scripts/gen_archive.py > ARCHIVE.md
```
