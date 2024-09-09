# Copyright (c) 2024, Aviv Bick, Kevin Li.

import argparse
import time
from functools import partial

import torch
from mamba_ssm import MambaLMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.lm_head import LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="The quick brown fox jumps over the lazy dog.")
    parser.add_argument("--promptlen", type=int, default=100)
    parser.add_argument(
        "--model",
        type=str,
        default="hybrid-phi-mamba",
        choices=["mamba1", "mamba2", "phi", "phi-mamba", "hybrid-phi-mamba"],
    )
    parser.add_argument("--genlen", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--mixed_precision", action="store_true")
    # Sampling arguments
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    return parser.parse_args()


@torch.inference_mode()
def time_bench(args, input_ids, generate_fn):
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
        for _ in range(args.repeats):
            out = generate_fn(
                input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen
            )
    torch.cuda.synchronize()

    # Print stats
    print(f"\nTiming results for {args.model} model:")
    print(
        f"Prompt length: {len(input_ids[0])}, generation length: {len(out.sequences[0]) - len(input_ids[0])}"
    )
    print(
        f"prompt processing + decoding time: {(time.time() - start) / args.repeats * 1000:.0f}ms"
    )


def choose_model(args):
    name = args.model
    assert name in ["mamba1", "mamba2", "phi", "phi-mamba", "hybrid-phi-mamba"]

    # Load model
    if name == "phi-mamba":
        model = LMHeadModel.from_pretrained("goombalab/Phi-Mamba", strict=True)
    elif name == "hybrid-phi-mamba":
        model = LMHeadModel.from_pretrained(
            "goombalab/Hybrid-Phi-Mamba",
            attn_type="flash_attention_2" if args.mixed_precision else "eager",
            strict=True,
        )
    elif name == "mamba1":
        model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-1.4b")
    elif name == "mamba2":
        model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-1.3b")
    elif name == "phi":
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")

    # Load tokenizer
    if name in ["mamba1", "mamba2"]:
        tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    elif name in ["phi", "phi-mamba", "hybrid-phi-mamba"]:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

    return model, tokenizer


@torch.inference_mode()
def main():

    # Parse arguments
    args = parse_args()
    torch.manual_seed(args.seed)

    # Load model
    model, tokenizer = choose_model(args)

    # Prepare model
    model.to(device=device)
    model.eval()
    print(
        f"Number of parameters: {sum(p.numel() for p in model.parameters())}"
    )

    # Tokenize prompt
    if args.prompt is None:
        input_ids = torch.randint(
            1, 1000, (1, args.promptlen), dtype=torch.long, device="cuda"
        )
        attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
    else:
        tokens = tokenizer(args.prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        attn_mask = tokens.attention_mask.to(device=device)

    # Prepare generation function
    if args.model == "phi":
        generate_fn = partial(
            model.generate,
            input_ids=input_ids,
            attention_mask=attn_mask,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    else:
        generate_fn = partial(
            model.generate,
            cg=(args.model in ["mamba1", "mamba2", "phi-mamba", "hybrid-phi-mamba"]),
            return_dict_in_generate=True,
            output_scores=False,
            enable_timing=False,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Generate
    with torch.cuda.amp.autocast(enabled=args.mixed_precision, dtype=torch.bfloat16):
        out = generate_fn(
            input_ids=input_ids, max_length=input_ids.shape[1] + args.genlen
        )

    if args.prompt is not None:
        print(
            "Generated text:\n",
            tokenizer.batch_decode(
                sequences=out.sequences.tolist(), 
                skip_special_tokens=True
            )[0],
        )

    time_bench(args, input_ids, generate_fn)


if __name__ == "__main__":
    main()
