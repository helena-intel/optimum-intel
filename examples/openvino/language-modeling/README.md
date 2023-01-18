# Language Modeling 

> **Note**
> This is work in progress and does not currently work

This folder contains [`run_clm.py`](https://github.com/huggingface/optimum/blob/main/examples/openvino/language-modeling/run_clm.py), a script to fine-tune a 🤗 Transformers model on a language modeling dataset while applying quantization aware training (QAT). QAT can be easily applied by replacing the Transformers [`Trainer`](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer) with the Optimum [`OVTrainer`].

**Note:** This script only works with models that have a fast tokenizer (backed by the 🤗 Tokenizers library) as it
uses special features of those tokenizers. You can check if your favorite model has a fast tokenizer in
[this table](https://huggingface.co/transformers/index.html#supported-frameworks).

## Fine-tuning WikiText

```bash
python run_clm.py 
  --model_name_or_path gpt2 
  --do_train 
  --do_eval 
  --dataset_name wikitext 
  --dataset_config wikitext-103-raw-v1 
  --num_train_epochs 3 
  --output_dir gpt2_wikitext2_int8 
  --per_gpu_eval_batch_size=1 
  --per_gpu_train_batch_size=2 
  --save_steps=591 
```
