from datasets import load_dataset


def main():
    dataset = load_dataset("vukrosic/blueberry-1B-pretrain", split="train")
    dataset.save_to_disk("processed_data/pretrain_1B")
    print("Saved processed_data/pretrain_1B")


if __name__ == "__main__":
    main()
