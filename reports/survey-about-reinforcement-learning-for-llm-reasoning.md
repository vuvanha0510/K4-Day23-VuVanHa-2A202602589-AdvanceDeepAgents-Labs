# A Survey on Reinforcement Learning for Large Language Model Reasoning

## TL;DR
- Reinforcement learning (RL) has driven a paradigm shift in Large Language Model (LLM) reasoning, transitioning from static supervised fine-tuning (SFT) to active reward optimization, test-time compute scaling, and long Chain-of-Thought (CoT) generation [1].
- Foundational RL algorithms have evolved from traditional actor-critic PPO to critic-free methods like Group Relative Policy Optimization (GRPO) and DAPO, which dramatically reduce memory overhead and stabilize training for complex reasoning tasks [2][3].
- Reward modeling has progressed from coarse Outcome Reward Models (ORMs) to fine-grained Process Reward Models (PRMs) and verifiable rule-based rewards (RLVR), with recent findings showing that outcome supervision can implicitly induce robust PRM capabilities [4][5][6].
- Frontier open-weights reasoning paradigms—exemplified by DeepSeek-R1 and QwQ-32B—demonstrate that pure or multi-stage RL from base models can spontaneously trigger self-reflection, verification, and performance competitive with proprietary models like OpenAI o1 [7][8].
- Key open challenges include mitigating reward hacking and repetition loops, reducing generation latency and token overhead, and optimizing compute allocation between inference-time search and model parameter scaling [4][8][6].

## Background
The quest for advanced reasoning in Large Language Models (LLMs)—spanning mathematics, coding, and multi-step symbolic problem solving—has increasingly shifted from traditional static instruction tuning toward reinforcement learning (RL) and reward-driven post-training. Historically, LLM alignment relied heavily on Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO) alongside learned scalar value critics [1][9]. However, standard PPO architectures encounter significant computational bottlenecks and training instability when applied to long-horizon, multi-step reasoning traces. 

To address these limitations, recent research has explored specialized RL algorithms, advanced reward design, test-time search guidance, and pure RL frameworks from foundation models. This survey synthesizes recent literature from 2024 to 2026, examining foundational algorithms, process supervision versus outcome rewards, frontier open-weights reasoning models such as DeepSeek-R1 and QwQ, and ongoing challenges in reasoning optimization.

## Foundational RL Algorithms and Critic-Free Optimization
The transition from general conversational alignment to mathematical and logical reasoning required rethinking policy optimization algorithms. Traditional RLHF frameworks relied on PPO paired with a separate value network to estimate the baseline value function. However, for large generative models engaged in multi-step reasoning, storing and updating the value network doubles memory requirements and introduces severe gradient variance.

This challenge led to the widespread adoption of critic-free and group-based policy gradient methods, most notably Group Relative Policy Optimization (GRPO) and Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) [2][3]. Introduced in DeepSeekMath, GRPO discards the value critic entirely and computes the baseline for a generated output by sampling a group of outputs for the same prompt and comparing their relative rewards [2][3]. By averaging rewards within the group, GRPO eliminates value network memory overhead while maintaining robust policy updates. For instance, DeepSeekMath-RL 7B utilized GRPO with group size $G = 64$, learning rate $1e-6$, and KL penalty $\beta = 0.04$, achieving 88.2% on GSM8K and 51.7% on MATH [3].

Further refinement of group-based optimization is embodied in DAPO, which addresses sample-level loss aggregation limitations in GRPO (such as brevity-induced reward hacking) through token-level formulations [2]. Using Qwen2.5-1.5B, DAPO reached 53.3% on GSM8K and 30.0% on MMLU-Pro [2]. Alongside direct policy gradients, implicit reward shaping via Direct Preference Optimization (DPO) and step-level preference techniques (such as Step-DPO) provide label-free or contrastive supervision without full online RL loops [1][4].

## Reward Modeling: Outcome Supervision vs. Process Supervision
Designing effective reward signals is critical for steering LLM reasoning. Reward modeling paradigms are broadly categorized into model-based rewards (Outcome Reward Models [ORMs] and Process Reward Models [PRMs]), rule-based verifiable rewards (RLVR), and implicit preference losses [1][4].

Outcome Reward Models evaluate only the final answer (binary or scalar correctness), whereas Process Reward Models provide granular verification across intermediate reasoning steps. PRMs enable precise error localization and effective guidance during search-based generation [4][10]. Granularity-regulated adaptive compute frameworks have demonstrated that optimally allocating compute between coarse ORMs and fine-grained PRMs significantly boosts test-time verification accuracy [10]. 

However, obtaining human-annotated step-level rewards is expensive. Recent empirical breakthroughs demonstrate that outcome-based reinforcement learning—using rule-based verifiers such as math correctness checkers and programming language execution servers—can implicitly induce and enhance PRM capabilities within LLMs without requiring explicit step-by-step supervision [8][6]. Furthermore, verifiable rule-based rewards (RLVR) prevent reward hacking and effectively stabilize long-horizon reasoning training [1][4].

## Test-Time Scaling and Search-Based Reinforcement Learning
Beyond parametric model updates, reasoning performance is heavily amplified by test-time compute scaling and search-based inference algorithms. Techniques such as Monte Carlo Tree Search (MCTS), Tree-of-Thought, and hierarchical retrieval-augmented frameworks (e.g., R2-LLMs) leverage PRMs to systematically explore, evaluate, and prune reasoning trajectories at inference time [11][12].

For instance, R2-LLM integrates dual-level retrieval-augmented in-context learning with MCTS guided by process reward models, significantly improving generalization on complex problem-solving benchmarks [11]. Similarly, training-free frameworks like SolverLLM leverage test-time scaling and structured search to translate optimization specifications into solver-ready code [13].

Crucially, compute-optimal inference scaling laws reveal that smaller models equipped with efficient test-time search can rival or outperform massive foundation models [14]. Efficient Tree Search (ETS) algorithms reduce memory bottlenecks and increase throughput by dynamically pruning redundant trajectories while preserving path diversity [14][7].

## Frontier Paradigms: DeepSeek-R1, QwQ, and Open-Weights Reasoning Models
The release of DeepSeek-R1 and Alibaba Cloud's QwQ-32B marked a watershed moment in open-weights reasoning models, demonstrating performance competitive with proprietary systems like OpenAI o1 [8][15]. 

DeepSeek-R1 demonstrated that advanced reasoning behaviors—such as self-reflection, backtracking, verification, and long Chain-of-Thought generation—can be spontaneously incentivized through pure reinforcement learning starting from a base model (DeepSeek-R1-Zero), eliminating the need for human-annotated reasoning trajectories [8]. To resolve issues such as readability degradation and language mixing observed in raw pure RL, DeepSeek-R1 implemented a multi-stage training pipeline combining a cold-start stage with thousands of long CoT examples, reasoning-oriented RL using GRPO, rejection sampling, and final alignment [8]. Furthermore, open-sourced distilled dense models (such as DeepSeek-R1-Distill-Qwen-32B) achieved 72.6% on AIME 2024 and 94.3% on MATH-500, outperforming earlier proprietary baselines [8].

Similarly, QwQ-32B utilized continuous reinforcement learning scaling atop robust foundation models using accuracy verifiers and execution servers, achieving state-of-the-art performance on math and coding benchmarks while maintaining strong instruction-following capabilities [15]. Practitioners emphasize specific decoding requirements for these models—such as temperature 0.6, TopP 0.95, and avoiding greedy decoding—to prevent endless repetition loops during extended reasoning [15].

## Trends and Open Problems
Despite rapid progress, reinforcement learning for LLM reasoning faces several critical open challenges:
- **Reward Hacking and Length Exploitation:** Group-based and outcome-driven reward optimization can inadvertently incentivize verbosity, repetition loops, or superficial formatting tricks rather than genuine logical rigor [4][2].
- **Token Generation Overhead:** Long Chain-of-Thought reasoning dramatically increases inference latency and computational costs per query, necessitating more efficient decoding and compressed reasoning representations [8][6].
- **Data Efficiency and Stability:** Large-scale online RL training remains sensitive to hyperparameter configurations (such as learning rates and KL penalty coefficients) and distribution shifts during multi-step rollout generation [4][3].
- **Generalization Beyond Domains:** While math and coding benefit greatly from rule-based verifiers (RLVR), translating verifiable rewards to open-ended creative tasks, commonsense reasoning, and multimodal domains remains an active area of research [10][5].

## References
[1] A Survey on Reward Models and Learning Strategies for Large Language Models. arxiv. https://arxiv.org/abs/2505.02686 (2025-05-02)
[2] A Systematic Comparison of PPO, GRPO, and DAPO for Reinforcement Learning in LLMs. arxiv. https://arxiv.org/abs/2512.07611 (2025-12-07)
[3] DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arxiv. https://arxiv.org/abs/2402.03300 (2024-04-27)
[4] Reward Modeling for Reinforcement Learning-Based LLM Reasoning. arxiv. https://arxiv.org/abs/2602.09305 (2026-02-09)
[5] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arxiv. https://arxiv.org/abs/2501.12948 (2025-01-22)
[6] Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs. arxiv. https://arxiv.org/abs/2505.11227 (2025-05-16)
[7] Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning. hf-daily. https://huggingface.co/papers/2509.23250 (2025-09-27)
[8] QwQ-32B: Embracing the Power of Reinforcement Learning. web. https://qwenlm.github.io/blog/qwq-32b/ (2025-03-06)
[9] RLHF Algorithms Ranked: An Extensive Evaluation Across Diverse Tasks, Rewards, and Hyperparameters. web. https://aclanthology.org/2025.emnlp-industry.35.pdf (N/A)
[10] Granularity-Regulated Adaptive Computational Efficiency for Optimal Verification in Test-Time Scaling. arxiv. https://arxiv.org/abs/2606.19354 (2026-04-28)
[11] Enhancing Test-Time Scaling of Large Language Models with Hierarchical Retrieval-Augmented MCTS (R2-LLMs). arxiv. https://arxiv.org/abs/2507.05557 (2025-07-08)
[12] SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search. arxiv. https://arxiv.org/abs/2510.16916 (2025-10-19)
[13] Efficient Tree Search for Inference-Time Scaling (ETS). hf-daily. https://huggingface.co/papers/2502.13575 (2025-02-19)
[14] Can 1B LLM Surpass 405B LLM? Rethinking Compute-Optimal Test-Time Scaling. hf-daily. https://huggingface.co/papers/2502.06703 (2025-02-10)
[15] Understanding R1-Zero-Like Training: A Critical Perspective. hf-daily. https://huggingface.co/papers/2503.20783 (2025-03-26)
