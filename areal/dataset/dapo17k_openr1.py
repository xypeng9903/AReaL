from datasets import load_dataset


def get_dapo17k_rl_dataset(
    path: str,
    tokenizer,
    max_length: int | None = None,
):
    dataset = load_dataset(path=path, name="all", split="train")

    def process(sample):
        prompt = sample["prompt"] + "\nPlease put your final answer within \\boxed{}."
        messages = [
            {"role": "user", "content": prompt}
        ]
        answer = sample["solution"]
        return {"messages": messages, "answer": answer}

    dataset = dataset.map(process)
    dataset = dataset.select_columns(["messages", "answer"])

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
    data_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/pengxinyu05/huggingface.co/datasets/open-r1/DAPO-Math-17k-Processed"
    
    tokenizer = load_hf_tokenizer(tokenizer_path)
    dataset = get_dapo17k_rl_dataset(data_path, tokenizer)

    print(f"{len(dataset)=}")
    
    print()
    print(dataset[0])
    print()

    print()
    print(dataset[1])
    print()