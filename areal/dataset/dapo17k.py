from datasets import load_dataset


def get_dapo17k_rl_dataset(
    path: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset("parquet", data_files=path, split="train")

    def process(sample):
        prompt = sample["prompt"][0]["content"]
        prompt = prompt.replace("Remember to put your answer on its own line after \"Answer:\"", "Please put your final answer within \\boxed{}")
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        answer = sample["reward_model"]["ground_truth"]
        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process).remove_columns(["prompt", "data_source", "ability", "reward_model"])

    # Filter out sequences longer than max_length if tokenizer and max_length are provided
    if max_length is not None:

        def filter_length(sample):
            # Tokenize the user content to check length
            content = sample["messages"][0]["content"]
            tokens = tokenizer.encode(content)
            return len(tokens) <= max_length

        dataset = dataset.filter(filter_length)

    return dataset


if __name__ == "__main__":
    from areal.utils.hf_utils import load_hf_tokenizer
    
    tokenizer_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/pengxinyu05/huggingface.co/Qwen/Qwen3-1.7B-Base"
    data_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/pengxinyu05/data/dapo-math-17k.parquet"
    
    tokenizer = load_hf_tokenizer(tokenizer_path)
    dataset = get_dapo17k_rl_dataset(data_path, tokenizer)

    print(f"{len(dataset)=}")
    
    print()
    print(dataset[0])
    print()

    print()
    print(dataset[1])
    print()