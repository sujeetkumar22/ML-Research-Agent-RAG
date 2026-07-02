"""
src/evaluation/evaluator.py

RAGAS-based evaluation pipeline.
This is what puts your project in the top 5% of portfolios.

Run this AFTER building the full RAG pipeline to get your real numbers.
The results go directly into your resume bullets.
"""
import json
from pathlib import Path
from loguru import logger


# Golden test set — 25 questions with ground truth answers
# Based on landmark papers in the ArXiv collection
GOLDEN_QA_PAIRS = [
    {
        "question": "What is the core innovation in the Transformer architecture introduced in 'Attention Is All You Need'?",
        "ground_truth": "The Transformer replaces recurrence and convolutions entirely with self-attention mechanisms. The key innovation is multi-head self-attention, which allows the model to attend to information from different representation subspaces at different positions simultaneously. This enables parallelization across sequence positions, unlike RNNs which process tokens sequentially."
    },
    {
        "question": "How does LoRA (Low-Rank Adaptation) reduce the number of trainable parameters during fine-tuning?",
        "ground_truth": "LoRA freezes the pre-trained model weights and injects trainable low-rank decomposition matrices into each layer of the Transformer architecture. Instead of fine-tuning all parameters, LoRA represents weight updates as the product of two smaller matrices (A and B), where the rank r is much smaller than the original weight dimensions. This reduces the number of trainable parameters by up to 10,000x compared to full fine-tuning."
    },
    {
        "question": "What are the key differences between BERT and GPT in terms of architecture and training objectives?",
        "ground_truth": "BERT uses a bidirectional encoder trained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). GPT uses a unidirectional (left-to-right) decoder trained with standard language modeling (predicting the next token). BERT is better for understanding tasks (classification, NER) while GPT is better for generation tasks. BERT can see context from both directions; GPT only sees preceding context."
    },
    {
        "question": "How does Retrieval-Augmented Generation (RAG) improve upon standard language model generation?",
        "ground_truth": "RAG combines a parametric memory (the language model) with a non-parametric memory (a retrieved document store). When generating an answer, RAG first retrieves relevant documents using a dense retrieval model (DPR), then conditions the generation on both the input query and the retrieved documents. This allows the model to access up-to-date information, reduces hallucination, and improves factual accuracy without requiring full model retraining."
    },
    {
        "question": "What is the attention formula used in the Transformer and why is the scaling factor important?",
        "ground_truth": "The attention formula is: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V, where Q, K, V are query, key, and value matrices and d_k is the key dimension. The scaling factor 1/sqrt(d_k) prevents the dot products from growing too large in magnitude, which would push the softmax function into regions with extremely small gradients (vanishing gradient problem). Without scaling, the softmax output would become very peaked, effectively attending to only one position."
    },
    {
        "question": "What is chain-of-thought prompting and when does it improve LLM performance?",
        "ground_truth": "Chain-of-thought (CoT) prompting is a technique where the model is prompted to generate intermediate reasoning steps before producing the final answer, using few-shot examples that show this reasoning process. CoT improves performance on tasks requiring multi-step reasoning, mathematical problem solving, and commonsense reasoning. It is most effective for larger models (100B+ parameters) and tasks that benefit from decomposition into sub-problems."
    },
    {
        "question": "How does CLIP learn visual representations and what makes it useful for zero-shot classification?",
        "ground_truth": "CLIP (Contrastive Language-Image Pre-training) trains a visual encoder and text encoder jointly using contrastive learning on 400 million image-text pairs from the internet. The model learns to maximize the cosine similarity of matching image-text pairs while minimizing it for non-matching pairs. For zero-shot classification, CLIP embeds class names as text descriptions and compares them to the image embedding — the class with highest similarity is predicted, enabling classification on unseen categories."
    },
    {
        "question": "What problem does batch normalization solve and how does it work?",
        "ground_truth": "Batch normalization addresses internal covariate shift — the change in the distribution of network activations as the parameters of the preceding layers change during training. It normalizes the inputs to each layer to have zero mean and unit variance across the mini-batch, then applies learnable scale (gamma) and shift (beta) parameters. This stabilizes training, allows higher learning rates, reduces sensitivity to initialization, and acts as a regularizer."
    },
    {
        "question": "What is QLoRA and how does it extend LoRA?",
        "ground_truth": "QLoRA (Quantized LoRA) extends LoRA by fine-tuning quantized models. It uses 4-bit NormalFloat (NF4) quantization for the frozen base model weights, double quantization to reduce memory further, and paged optimizers to manage memory spikes. This allows fine-tuning of 65B parameter models on a single 48GB GPU, whereas standard LoRA would require much more memory. QLoRA maintains full 16-bit fine-tuning performance while dramatically reducing GPU memory requirements."
    },
    {
        "question": "What is RLHF and how is it used to align language models?",
        "ground_truth": "RLHF (Reinforcement Learning from Human Feedback) aligns LLMs with human preferences through three steps: (1) Supervised fine-tuning on high-quality demonstration data, (2) Training a reward model on human preference comparisons between model outputs, (3) Fine-tuning the LLM using PPO (Proximal Policy Optimization) to maximize reward model scores while adding a KL divergence penalty to prevent the model from deviating too far from the SFT model. InstructGPT uses RLHF and demonstrates that smaller RLHF-tuned models can outperform much larger unaligned models."
    },
    {
        "question": "How does the Adam optimizer work and why is it preferred over standard SGD?",
        "ground_truth": "Adam (Adaptive Moment Estimation) maintains per-parameter adaptive learning rates. It keeps exponential moving averages of both the gradients (first moment) and squared gradients (second moment), with hyperparameters beta1 and beta2 controlling their decay rates. The update rule normalizes gradients by the square root of the second moment, giving larger updates to parameters with infrequent updates and smaller updates to those with frequent updates. Adam is preferred over SGD because it requires less learning rate tuning, handles sparse gradients well, and converges faster on most deep learning tasks."
    },
    {
        "question": "What is the Vision Transformer (ViT) and how does it apply transformers to images?",
        "ground_truth": "ViT (Vision Transformer) applies a standard Transformer encoder to image classification by splitting images into fixed-size patches (e.g., 16x16 pixels), linearly projecting each patch, and treating the sequence of patch embeddings as tokens — similar to word tokens in NLP. Position embeddings are added to preserve spatial information. A [CLS] token is prepended whose final hidden state is used for classification. ViT requires large-scale pretraining data to match CNN performance, but scales better than CNNs with more data and compute."
    },
    {
        "question": "What is Sentence-BERT and how does it improve upon BERT for sentence similarity?",
        "ground_truth": "Sentence-BERT (SBERT) modifies the pretrained BERT network using siamese and triplet network structures to derive semantically meaningful sentence embeddings. Unlike BERT, which requires feeding both sentences as a pair (O(n^2) complexity for finding similar sentences in a corpus), SBERT produces fixed-size sentence embeddings that can be compared using cosine similarity in O(n) time. It is fine-tuned using natural language inference (NLI) data and semantic textual similarity (STS) datasets."
    },
    {
        "question": "What is Flash Attention and why is it faster than standard attention?",
        "ground_truth": "Flash Attention is an IO-aware exact attention algorithm that is 2-4x faster than standard attention without approximation. Standard attention materializes the full N×N attention matrix in GPU HBM (high bandwidth memory), which is slow due to memory bandwidth bottlenecks. Flash Attention uses tiling to break the computation into blocks that fit in SRAM (fast on-chip memory), computing attention block by block and never materializing the full attention matrix in HBM. It also fuses the attention computation into a single GPU kernel, reducing memory accesses."
    },
    {
        "question": "What is dropout and how does it prevent overfitting?",
        "ground_truth": "Dropout is a regularization technique where, during training, each neuron's output is independently set to zero with probability p (the dropout rate, typically 0.1-0.5). This prevents neurons from co-adapting — no single neuron can rely on the presence of specific other neurons. At test time, all neurons are active but outputs are scaled by (1-p) to maintain the same expected activation magnitude. Dropout can be interpreted as training an ensemble of 2^n different neural networks sharing weights, where n is the number of neurons."
    },
    {
        "question": "How does ResNet solve the vanishing gradient problem in deep networks?",
        "ground_truth": "ResNet introduces skip connections (also called residual connections or shortcut connections) that add the input of a layer block directly to its output: output = F(x) + x, where F(x) is the learned residual mapping. If the optimal function is close to an identity mapping, it is easier to push F(x) toward zero than to learn the identity function directly. During backpropagation, gradients can flow directly through the skip connection without passing through multiple non-linearities, preventing vanishing gradients in very deep networks (100+ layers)."
    },
    {
        "question": "What is the difference between encoder-only, decoder-only, and encoder-decoder Transformer architectures?",
        "ground_truth": "Encoder-only models (BERT) process the full input bidirectionally and produce contextualized representations — best for understanding tasks like classification, NER, and extractive QA. Decoder-only models (GPT) generate text autoregressively using causal (left-to-right) attention — best for generation tasks. Encoder-decoder models (T5, BART) use the encoder to process input and the decoder to generate output with cross-attention to encoder states — best for sequence-to-sequence tasks like translation, summarization, and abstractive QA."
    },
    {
        "question": "What is Direct Preference Optimization (DPO) and how does it differ from RLHF?",
        "ground_truth": "DPO (Direct Preference Optimization) is a simpler alternative to RLHF that trains language models directly on preference data without requiring a separate reward model or RL optimization. While RLHF requires training a reward model and then running PPO with a KL penalty, DPO re-parameterizes the reward function in terms of the optimal policy and solves the constrained RL problem analytically, resulting in a simple classification loss on preferred vs rejected responses. DPO is more stable, requires less compute, and avoids the reward model training step."
    },
    {
        "question": "What is XGBoost and what makes it effective for tabular data?",
        "ground_truth": "XGBoost (Extreme Gradient Boosting) is an optimized distributed gradient boosting library. It trains an ensemble of decision trees sequentially, where each new tree fits the residual errors of the current ensemble using second-order gradient statistics (both gradient and Hessian). Key features: regularization terms in the objective (L1 and L2) to prevent overfitting, efficient handling of missing values, column subsampling, tree pruning via depth-first strategy with gain-based pruning, and parallel computation of split finding using quantile sketching. It excels on tabular data due to these regularization techniques and its ability to model complex non-linear interactions."
    },
    {
        "question": "How does Stable Diffusion generate images from text prompts?",
        "ground_truth": "Stable Diffusion is a latent diffusion model (LDM) that performs the diffusion process in a compressed latent space rather than pixel space. It uses a VAE to encode images into a lower-dimensional latent representation, trains a U-Net denoising model in this latent space, and uses a CLIP text encoder to condition the denoising on text prompts via cross-attention. At inference, it starts from Gaussian noise in latent space, iteratively denoises it conditioned on the text embedding, then decodes to an image with the VAE decoder. Operating in latent space is 4-8x more efficient than pixel-space diffusion."
    },
    {
        "question": "What are the key contributions of the LLaMA model family?",
        "ground_truth": "LLaMA demonstrated that smaller, highly optimized models trained on more data can outperform larger models trained on less data. Key contributions: trained exclusively on publicly available data (The Pile, Common Crawl, Wikipedia), showed that a 13B parameter model can match GPT-3 (175B) on most benchmarks by training with more compute-optimal token counts, used architectural improvements like RoPE (Rotary Position Embeddings), SwiGLU activation, and RMS normalization. LLaMA 2 added supervised fine-tuning and RLHF alignment, group query attention for efficiency, and a longer 4096 context window."
    },
    {
        "question": "What is BERT's Masked Language Modeling (MLM) pre-training objective?",
        "ground_truth": "In MLM, 15% of tokens in the input are randomly selected for masking. Of these selected tokens: 80% are replaced with a [MASK] token, 10% are replaced with a random token, and 10% are kept unchanged. The model must predict the original tokens at all masked positions. This bidirectional training objective allows BERT to build deep contextual representations using both left and right context, unlike GPT which only uses left context. The 10% random replacement helps the model be robust and the 10% unchanged helps with the discrepancy between pretraining and fine-tuning."
    },
    {
        "question": "What is DistilBERT and what is the knowledge distillation approach it uses?",
        "ground_truth": "DistilBERT is a smaller, faster, and lighter version of BERT trained using knowledge distillation. It has 40% fewer parameters and runs 60% faster than BERT-base while retaining 97% of BERT's performance on NLU tasks. Knowledge distillation trains the smaller 'student' model to mimic the outputs of the larger 'teacher' BERT, using a combination of three loss functions: distillation loss (KL divergence between student and teacher soft probability distributions), masked language modeling loss, and cosine embedding loss to align hidden state vectors between student and teacher."
    },
    {
        "question": "What is the ReAct framework for LLM agents?",
        "ground_truth": "ReAct (Reasoning + Acting) synergizes reasoning and acting in LLMs by generating both verbal reasoning traces and task-specific actions in an interleaved manner. The model alternates between Thought (internal reasoning about the current state and what to do next), Action (calling an external tool like a search engine or calculator), and Observation (the result returned by the tool). This interleaving allows the model to use reasoning to guide actions, use observations to update reasoning, and create a more interpretable trajectory. ReAct significantly outperforms chain-of-thought alone on knowledge-intensive tasks requiring external information."
    },
    {
        "question": "How does word2vec learn word embeddings?",
        "ground_truth": "Word2vec learns embeddings by training a shallow neural network on one of two tasks: Skip-gram (predicts surrounding context words given a target word) or CBOW (Continuous Bag of Words, predicts a target word given surrounding context words). The learned hidden layer weights become the word embeddings. The key insight is that words with similar contexts will have similar embeddings, resulting in a vector space where semantic relationships are encoded as linear directions (e.g., king - man + woman ≈ queen). Negative sampling is used instead of full softmax for efficiency."
    },
]


def run_evaluation(
    retriever,
    generator,
    qa_pairs: list[dict] = None,
    output_path: str = "data/processed/eval_results.json",
) -> dict:
    """
    Run RAGAS evaluation on the golden test set.

    This is the function that produces your resume numbers.
    Run it at the end once your pipeline is working.

    Returns dict with metric scores.
    """
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset
    except ImportError:
        logger.error("RAGAS not installed. Run: pip install ragas datasets")
        return {}

    if qa_pairs is None:
        qa_pairs = GOLDEN_QA_PAIRS

    logger.info(f"Running RAGAS evaluation on {len(qa_pairs)} questions…")

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, qa in enumerate(qa_pairs, 1):
        logger.info(f"[{i}/{len(qa_pairs)}] {qa['question'][:60]}…")
        try:
            # Retrieve
            chunks = retriever.retrieve(qa["question"], top_k=5)
            contexts = [c["text"] for c in chunks]

            # Generate (non-streaming for eval)
            answer = generator.generate(
                qa["question"], chunks, stream=False
            )

            data["question"].append(qa["question"])
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(qa["ground_truth"])

        except Exception as e:
            logger.error(f"Error on Q{i}: {e}")
            continue

    if not data["question"]:
        logger.error("No questions evaluated successfully")
        return {}

    dataset = Dataset.from_dict(data)
    result = ragas_evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
    )

    scores = {
        "faithfulness":       round(float(result["faithfulness"]), 3),
        "answer_relevancy":   round(float(result["answer_relevancy"]), 3),
        "context_recall":     round(float(result["context_recall"]), 3),
        "context_precision":  round(float(result["context_precision"]), 3),
        "num_questions":      len(data["question"]),
    }

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scores, f, indent=2)

    logger.success(f"\n{'='*50}")
    logger.success("RAGAS EVALUATION RESULTS")
    logger.success(f"{'='*50}")
    for k, v in scores.items():
        logger.success(f"  {k:<25} {v}")
    logger.success(f"{'='*50}")
    logger.success(f"Results saved to {output_path}")

    return scores
