# Comprehensive Survey on Video and Multimodal Generation

## TL;DR
- Video generation models have transitioned from restricted text-to-image extensions to advanced diffusion transformers (DiTs) and autoregressive tokenizers capable of producing high-definition, minute-long, and real-time motion sequences [1][2][3].
- Open-source video generation frameworks like HunyuanVideo, Open-Sora, CogVideoX, and LTX-Video rival proprietary counterparts by utilizing decoupled spatial-temporal attention, 3D causal VAEs, and parameter scaling up to 13 billion parameters [4][5][3][6].
- Unified multimodal foundation models employ early-fusion tokenization (Chameleon) or discrete modality-specific codebooks (AnyGPT, VITA, SEED-X) to seamlessly interleave and generate text, audio, images, and video within a single architecture [7][8][9][10].
- Comprehensive benchmark suites such as VBench, VBench++, and PhyWorldBench expose ongoing limitations in physical realism, temporal consistency, and motion dynamics, driving innovations in sub-quadratic attention and uncertainty calibration [11][12][13].

## Background
The field of generative artificial intelligence has experienced a profound paradigm shift from static 2D image synthesis to dynamic video and comprehensive multimodal generation. Early generative models relied heavily on generative adversarial networks (GANs) and basic recurrent or 3D U-Net architectures, which struggled with temporal coherence, spatial resolution, and semantic alignment over extended durations. 

With the advent of diffusion probabilistic models and large-scale autoregressive transformers, the generative landscape evolved rapidly. Video generation introduces complex spatio-temporal dependencies, requiring models to maintain physical consistency, lighting continuity, and smooth motion across dozens or hundreds of frames. Concurrently, multimodal generation seeks to unify diverse modalities—text, audio, images, and video—into shared representation spaces. These developments bridge computer vision, natural language processing, and audio processing, creating foundations for general-purpose world simulators and interactive omni-modal assistants.

## Foundations of Video Generation: Architectures and Tokenization
Modern video generation models predominantly adopt two architectural paradigms: Diffusion Transformers (DiTs) and decoder-only autoregressive transformers. Sora pioneered the use of DiTs operating on spacetime latent patches extracted by spatial-temporal VAEs, which compress raw video inputs temporally and spatially [1]. This design enables flexible generation resolutions, aspect ratios, and durations up to one minute [1].

Similarly, Open-Sora introduces the Spatial-Temporal Diffusion Transformer (STDiT), which decouples spatial attention within frames from temporal attention across frames, initialized from PixArt-$\alpha$ and enhanced with rotary positional embeddings (RoPE) [4]. Emu Video factorizes text-to-video generation into explicit keyframe image generation followed by a latent diffusion U-Net conditioned on both text and the generated image, eliminating train-test distribution mismatches via zero terminal-SNR noise schedules [14].

In contrast, VideoPoet treats video generation through a prefix language model architecture that tokenizes multimodal data (text, video, audio) into unified discrete tokens using MAGVIT-v2 and SoundStream, bypassing continuous latent diffusion in favor of autoregressive next-token prediction [2].

## Open-Source Video Models and Scaling Frameworks
The open-source community has made monumental strides in democratizing high-performance video generation. HunyuanVideo emerges as a systematic framework exceeding 13 billion parameters, featuring an advanced DiT architecture and extensive training that competes with or surpasses leading proprietary models in visual quality and prompt adherence [5].

CogVideoX scales text-to-video diffusion transformer models to 2B and 5B parameters, capable of generating 10-second continuous videos at 16 fps and 768x1360 resolution using a 3D causal VAE and Expert Adaptive LayerNorm [6]. LTX-Video achieves faster-than-real-time generation (producing 5 seconds of 24fps video at 768x512 in 2 seconds on an H100 GPU) via a holistic 1:192 compression VAE and pixel-space denoising decoder [3]. Furthermore, specialized frameworks like MVDream leverage multi-view diffusion fine-tuned on 3D assets to generate consistent multi-view images from text prompts, serving as generalizable 3D priors for Score Distillation Sampling (SDS) [15].

## Unified Multimodal Generation: Early-Fusion and Discrete Codebooks
Beyond isolated video synthesis, unified multimodal generation aims to comprehend and generate arbitrary combinations of text, images, audio, and video. Chameleon adopts a fully token-based early-fusion foundation model approach, quantizing both text and images into discrete tokens within a shared vocabulary and training a dense transformer from scratch on interleaved mixed-modal sequences [7].

Modular and discrete codebook approaches are exemplified by AnyGPT, which utilizes modality-specific tokenizers (SpeechTokenizer, Encodec, VQ-codebooks) to compress speech, text, images, and music into discrete tokens processed by a frozen LLaMA-2 7B backbone [8]. VITA extends open-source interactive omni-multimodal LLMs based on Mixtral 8x7B to process video, image, text, and audio simultaneously using independent encoders connected via MLPs and special state tokens [9]. SEED-X integrates multi-granularity visual comprehension and generation using a pretrained ViT tokenizer combined with a multi-granularity de-tokenizer connected to an SD-XL U-Net via cross-attention layers [10]. Furthermore, Google Gemini established early foundational milestones for natively multimodal models trained jointly across image, audio, video, and text from the outset [16].

## Evaluation Benchmarks, Physical Realism, and Efficiency
Rigorous evaluation remains critical for tracking progress in video and multimodal generation. VBench and VBench++ establish comprehensive benchmark suites evaluating video models across 16 specific dimensions aligned with human perception, incorporating hierarchical dimensions and human preference annotations [11][12]. Video-ChatGPT introduces quantitative evaluation frameworks for video dialogue models across 5 core aspects: correctness, detail orientation, contextual understanding, temporal understanding, and consistency [17].

To address physical realism and hallucinations, PhyWorldBench benchmarks text-to-video models for adherence to physical laws through an "Anti-Physics" category, revealing significant gaps in gravity representation and object interaction [13]. On the efficiency front, heavy spatial-temporal computational bottlenecks in DiTs are tackled by innovations such as Sliding Tile Attention (STA) [18], sub-quadratic attention distillation pipelines (SQuad), and calibrated uncertainty quantification methods (C3) for controllable video generation.

## Trends and open problems
Current trajectories in video and multimodal generation point toward real-time interactive generation, native omni-modal integration, and rigorous physical world modeling. However, several open problems persist:
1. **Physical Plausibility**: Models frequently violate fundamental physical laws (gravity, momentum, occlusion), as highlighted by PhyWorldBench [13]. Developing intrinsic world physics priors remains an open challenge.
2. **Computational Complexity**: Spatio-temporal attention scales quadratically with resolution and duration. While sliding tile attention and distillation offer relief [18], achieving high-definition, long-form video synthesis at interactive frame rates remains resource-intensive.
3. **Evaluation Alignment**: Automated metrics often diverge from human perceptual preferences. Advancing multidimensional benchmarks (VBench++, C3) [12] with robust uncertainty estimation is essential for reliable deployment.
4. **Multimodal Alignment & Hallucination**: Unified early-fusion and discrete codebook models can suffer from cross-modal interference and hallucination during open-ended generation tasks, necessitating advanced instruction-tuning and alignment protocols.

## References
[1] Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models. arxiv. https://arxiv.org/abs/2402.17177 (2024-02-27)
[2] VideoPoet: A Large Language Model for Zero-Shot Video Generation. arxiv. https://arxiv.org/abs/2312.14125 (2023-12-21)
[3] LTX-Video: Realtime Video Latent Diffusion. arxiv. https://arxiv.org/abs/2501.00103 (2024-12-30)
[4] Open-Sora: Democratizing Efficient Video Production for All. arxiv. https://arxiv.org/abs/2412.20404 (2024-12-23)
[5] HunyuanVideo: A Systematic Framework For Large Video Generative Models. hf-daily. https://huggingface.co/papers/2412.03603 (2024-12-03)
[6] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2024-12-10)
[7] Chameleon: Mixed-Modal Early-Fusion Foundation Models. arxiv. https://arxiv.org/abs/2405.09818 (2024-05-16)
[8] AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling. arxiv. https://arxiv.org/abs/2402.12226 (2024-02-20)
[9] VITA: Towards Open-Source Interactive Omni Multimodal LLM. arxiv. https://arxiv.org/abs/2408.05211 (2024-08-12)
[10] SEED-X: Multimodal Models with Unified Multi-granularity Comprehension and Generation. arxiv. https://arxiv.org/abs/2404.14396 (2024-04-23)
[11] VBench: Comprehensive Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2311.17982 (2023-11-29)
[12] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. hf-search. https://huggingface.co/papers/2411.13503 (2024-11-20)
[13] PhyWorldBench: A Comprehensive Evaluation of Physical Realism in Text-to-Video Models. hf-search. https://huggingface.co/papers/2507.13428 (2025-07-17)
[14] Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. arxiv. https://arxiv.org/abs/2311.10709 (2023-11-17)
[15] MVDream: Multi-view Diffusion for 3D Generation. arxiv. https://arxiv.org/abs/2308.16512 (2023-08-30)
[16] Gemini: A Family of Highly Capable Multimodal Models. arxiv. https://arxiv.org/abs/2312.11805 (2023-12-14)
[17] Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models. hf-search. https://huggingface.co/papers/2306.05424 (2024-06-10)
[18] Fast Video Generation with Sliding Tile Attention. hf-search. https://huggingface.co/papers/2502.04507 (2025-02-06)
