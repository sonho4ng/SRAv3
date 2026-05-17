from evaluator import Evaluator
import torch


evaluator = Evaluator(
    tokenizer_path=args.teacher_tokenizer,
    model_path=None,
    sft_lora=None,
    distilled_lora=None,
    seeds=[10]
)

# evaluator.model = trainer.student.model.model

benchmark_configs = {'dolly': './data/llm/dolly/valid.jsonl',
                     'self_instruct': './data/llm/self-inst/valid.jsonl',
                     'vicuna': './data/llm/vicuna/valid.jsonl',
                     'sni': './data/llm/sinst/11_/valid.jsonl',
                     'unni':'./data/llm/dialog/valid.jsonl'
                    }

with torch.cuda.amp.autocast(dtype=torch.float16):
    results = evaluator.evaluate_multiple_benchmarks(
        benchmark_configs=benchmark_configs,
        batch_size=32,
        max_seq_length=256,
        max_new_tokens=512
    )