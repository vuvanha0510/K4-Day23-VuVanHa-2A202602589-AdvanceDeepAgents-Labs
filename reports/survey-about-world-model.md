# Comprehensive Survey on World Models in Artificial Intelligence

## TL;DR
- World models serve as internal simulation engines that capture environmental dynamics to enable prediction, planning, and decision-making across reinforcement learning, robotics, and autonomous driving [1][2][3].
- Recent methodological paradigm shifts have moved world modeling beyond traditional recurrent state-space latent models [4] to massive Diffusion Transformers (DiTs) [5][6] and self-supervised Joint-Embedding Predictive Architectures (JEPAs) [7][8].
- While generative video models (such as Sora and Genie) achieve unprecedented visual realism and emergent 3D consistency [5][9][10], empirical benchmarks reveal a persistent "perception-function gap" where high visual fidelity does not guarantee closed-loop task success [11][12].
- Critical open challenges include compounding prediction errors over long horizons, physical law violations under distribution shift, weak action conditioning, and fragmented evaluation protocols [13][14][12].
- Emerging future directions emphasize hybrid neuro-symbolic integration, hierarchical active inference, and joint simulator-policy co-evolution to bridge the gap between perceptual simulation and robust physical control [15][16].

## Background
The intellectual lineage of world models spans cognitive science (Craik 1943; Miller et al. 1960), control theory (Conant and Ashby 1970; Bryson and Ho 1975), and classical robot planning before emerging as a core paradigm in modern machine learning [13]. In reinforcement learning, Ha and Schmidhuber (2018) established the canonical modular architecture separating spatial compression (Vision model $V$) from temporal transition modeling (MDN-RNN $M$), allowing agents to learn policies entirely within hallucinated dream environments [4]. 

Formally, a world model learns a transition function $\mathcal{P}(s_{t+1}|s_{t},a_{t})$ mapping current states $s_t$ and actions $a_t$ to future states $s_{t+1}$ [17]. Over the past decade, this formulation has expanded from low-dimensional control tasks to high-dimensional multi-modal settings encompassing video generation, autonomous driving, and embodied robotics [1][2][18]. Understanding these systems requires examining their multi-axis taxonomies, architectural paradigms, domain-specific instantiations, evaluation frameworks, and persistent open challenges [1][2][11][12].

## Foundations and Taxonomies of World Models
Recent comprehensive surveys categorize world models along multi-axis taxonomies that span architecture, methodologies, reasoning strategies, and application domains [1]. Architectural dimensions classify models by representation format (latent vectors, 3D occupancy grids, point clouds), dynamics formulation (recurrent state-space models, transformers, diffusion models), input modality (visual, language, 3D geometric), and learning paradigm (self-supervised, online model-based RL, offline batch learning, foundation pretraining) [1].

In parallel, agentic capability frameworks organize world models into a three-level hierarchy: L1 Predictors (one-step local transition operators), L2 Simulators (multi-step, action-conditioned rollouts respecting domain laws), and L3 Evolvers (systems that autonomously revise their internal models when predictions fail against new evidence) [2]. These capabilities operate across diverse governing-law regimes, including physical, digital, social, and scientific domains [2]. Historically, model-based RL frameworks (such as PlaNet and the Dreamer family) relied heavily on recurrent state-space models (RSSMs) to handle stochastic environments, whereas modern approaches increasingly leverage self-supervised representation learning and transformer backbones to capture complex spatial-temporal dynamics [1][3].

## Generative Video Models and Interactive Simulation Engines
A major transformation in world modeling has been driven by large-scale generative video models and interactive simulation engines [5][9][10]. Moving beyond simple latent transition models, architectures like Sora leverage Diffusion Transformers (DiTs) operating on spatiotemporal video patches compressed through a low-dimensional tokenizer [5]. These models exhibit emergent physical and digital world simulation capabilities at scale, including 3D consistency, long-range object permanence, and interactive modifications [5][9].

Similarly, foundational interactive environments like Genie employ an 11 billion parameter architecture trained on unlabelled internet videos without ground-truth action labels, utilizing a spatiotemporal video tokenizer, an autoregressive dynamics model, and a scalable latent action model to enable frame-by-frame user interaction [10]. In interactive gaming and open-world generation, GameGen-X employs a two-stage training paradigm combining foundation pretraining on OGameData with instruction tuning via InstructNet to integrate multimodal action control [6]. 

Complementary non-generative frameworks avoid pixel-level reconstruction overhead entirely. V-JEPA (Video Joint Embedding Predictive Architecture) and VJ-VCR perform self-supervised representation learning in abstract latent spaces using Vision Transformer encoders and variance-covariance regularization [7][8]. These joint-embedding architectures achieve superior training efficiency (1.5x to 6x faster) and robust motion understanding by predicting high-level feature dynamics rather than pixel-level details [7][8].

## Applications in Robotics and Autonomous Driving
World models serve as crucial predictive components in autonomous driving and embodied AI, supporting simulation, planning, and policy learning [13][18][19]. In autonomous driving, world models are classified across three tiers: generation of future physical physical worlds (image, BEV, occupancy grid, and point cloud forecasting), behavior planning for intelligent agents, and the tight coupling of prediction with motion planning [18]. Systems like DriveDreamer, GenAD, and DriveWorld utilize RSSM and BEV token conditioning to simulate rare edge cases and guide multi-agent collaborative decision-making [19].

In robotic manipulation and navigation, world models act as learned interactive simulators that reduce real-world sample complexity [13][20]. Systems such as DayDreamer and SWIM enable robot learning within hours of real-world interaction or minimal video fine-tuning [20]. However, empirical meta-analyses reveal a profound "perception-function gap" [11][12]: out of 160 surveyed benchmarks, 86% are model-agnostic, and very few establish direct VLA-versus-world-model contrasts or evaluate counterfactual execution [11]. While visual realism is high in models like Sora, their lack of native action sensitivity often limits their utility in closed-loop robotic control [14][11].

## Evaluation Benchmarks and Open Challenges
Evaluating world models remains a central bottleneck in the field due to fragmented protocols and lack of standardization [12]. Traditional metrics such as Peak Signal-to-Noise Ratio (PSNR), Structural Similarity (SSIM), and Fréchet Video Distance (FVD) measure low-level pixel reconstruction but fail to capture functional decision-making utility [19][12]. State-of-the-art assessments (e.g., WorldArena 2.0 and RoboWM-Bench) highlight structural limitations including compounding prediction errors over long horizons, state drift, physical law violations under out-of-distribution shifts, weak action conditioning, and hard sim-to-real transfer gaps [11][12]. Furthermore, Vision-Language Models (VLMs) evaluated on atomic physical reasoning exhibit severe limitations in spatial reasoning and future prediction [21].

## Trends and Open Problems
To overcome current limitations, research is increasingly converging on hybrid and structured architectures [15][16]. Promising future directions include deep hybrid active inference models that integrate discrete decision-making with continuous motion across temporal scales [15]. Additionally, neuro-symbolic agentic frameworks combine neural pattern recognition with symbolic logical reasoning, yielding major performance gains in embodied task completion and sample efficiency [16]. Another vital frontier is joint simulator-policy co-evolution (such as World-VLA-Loop), where world models iteratively refine their rollout reliability and adapt alongside downstream agent policies [13]. Addressing these challenges will require unified benchmarks that bridge perceptual fidelity with closed-loop functional control [11][12].

## References
[1] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications. arxiv. https://arxiv.org/abs/2606.00133 (2026-05-28)
[2] Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond. arxiv. https://arxiv.org/abs/2604.22748 (2026-04-22)
[3] World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications (Web Survey). web. https://arxiv.org/html/2606.00133 (2026-05-28)
[4] Recurrent World Models Facilitate Policy Evolution. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[5] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. hf-search. https://huggingface.co/papers/2402.17177 (2024-02-27)
[6] GameGen-X: Interactive Open-world Game Video Generation. hf-search. https://huggingface.co/papers/2411.00769 (2024-11-01)
[7] Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA). web. https://arxiv.org/abs/2404.08471 (2024-04-08)
[8] Video Representation Learning with Joint-Embedding Predictive Architectures (VJ-VCR). web. https://arxiv.org/abs/2412.10925 (2024-12-10)
[9] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. hf-search. https://huggingface.co/papers/2405.03520 (2024-05-06)
[10] Genie: Generative Interactive Environments. web. https://proceedings.mlr.press/v235/bruce24a.html (2024-07-08)
[11] Do World Models Make Better Robots?. arxiv. https://arxiv.org/abs/2609.29669 (2026-09-29)
[12] State of World Models 2026: Taxonomy, Benchmarks and Open Challenges. web. https://world-models.io/reports/state-of-world-models-2026/state-of-world-models-2026-v1.0.pdf (2026-01-01)
[13] World Model for Robot Learning: A Comprehensive Survey. arxiv. https://arxiv.org/abs/2605.00080 (2026-05-01)
[14] How Far is Video Generation from World Model: A Physical Law Perspective. hf-search. https://huggingface.co/papers/2411.02385 (2024-11-04)
[15] Deep Hybrid Models: Infer and Plan in a Dynamic World. web. https://www.mdpi.com/1099-4300/27/6/570 (2025-05-27)
[16] Neuro-symbolic Agentic AI: Architectures, Integration Patterns, Applications, Open Challenges and Future Research Directions. web. https://dl.acm.org/doi/10.1016/j.cosrev.2026.100902 (2026-06-24)
[17] Video Generation Models as World Models: Efficient Paradigms, Architectures, and Inference Algorithms. web. https://arxiv.org/abs/2603.28489 (2026-03-28)
[18] A Survey of World Models for Autonomous Driving. arxiv. https://arxiv.org/abs/2501.11260 (2025-01-18)
[19] A Comprehensive Survey on World Models for Embodied AI. hf-search. https://huggingface.co/papers/2510.16732 (2025-10-19)
[20] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[21] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation. hf-search. https://huggingface.co/papers/2506.21876 (2026-06-27)
