import torch
from tqdm import tqdm
import json
import re
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from prompts import system_prompt_kp, base_prompt_kp, system_prompt_kp_cmv, base_prompt_kp_cmv

def llm_inference(model, tokenizer, dataset, max_words, task="cmv"):
    """Run LLM inference to extract key points from clusters in the dataset."""
    final_outputs = []
    if task == "cmv":
        prompts = {
            "system": system_prompt_kp_cmv,
            "base": base_prompt_kp_cmv
        }
    else:
        prompts = {
            "system": system_prompt_kp,
            "base": base_prompt_kp
        }
    for topic, topic_data in tqdm(dataset.items()):
        for stance, stance_data in topic_data.items():
            print(f"Topic: {topic}, Stance: {stance}, Clusters: {len(stance_data)}")
            clusters = ""
            for cluter_id, cluster_data in stance_data.items():
                clusters += f"**Cluster {cluter_id}** \n- " + "\n- ".join([item["argument"] for item in cluster_data]) + "\n\n"
            system_prompt_text = prompts["system"].format(kp_token_length=max_words)
            text_stance = "supporting" if stance == "1" else "opposing"
            prompt_text = prompts["base"].format(
                kp_token_length=max_words,
                topic=topic,
                stance=text_stance,
                num_clusters=len(stance_data),
                clusters=clusters,
            )
            messages = [
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": prompt_text}
            ]
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                max_length=8192,
                pad_token_id=128001
            ).to("cuda")
            output = model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                use_cache=True,
                temperature=0.1,
                do_sample=True,
                top_p=0.94,
            )
            generated_tokens = output[0][inputs.shape[-1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            row = {
                "topic": topic,
                "stance": stance,
                "llm_output": output_text,
                "num_users": len(stance_data),
                "num_arguments": sum(len(cluster_data) for cluster_data in stance_data.values())
            }
            final_outputs.append(row)
    return final_outputs


def load_data(data_path, by_adu=False):
    """Load dataset from a JSON file."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def save_outputs(kp_outputs, test_dataset, output_path):
    """Save extracted key points and metadata to a JSON file."""
    final_output = []
    for item in kp_outputs:
        topic = item["topic"]
        stance = item["stance"]
        kps = re.findall(r"- (.*)", item["llm_output"])
        kps = [arg.strip() for arg in kps]
        print(f"Total Clusters: {len(test_dataset[topic][stance])}, Extracted KPs: {len(kps)}")
        final_output.append({
            "topic": topic,
            "stance": stance,
            "kps": kps,
            "llm_output": item["llm_output"],
            "num_users": item["num_users"],
            "num_arguments": item["num_arguments"],
        })
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=4)

def get_generation_pipeline(model_name, chat_template="llama-3.1"):
    """Load model and tokenizer for inference with specified chat template."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=8192,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
    tokenizer.padding = True
    tokenizer.truncation = True
    tokenizer.padding_side = "right"
    FastLanguageModel.for_inference(model)
    return model, tokenizer

   
def main(args):
    """Main function to run key point extraction pipeline."""
    print(f"Running kp extraction for {args.task}")
    test_dataset = load_data(args.file_path)
    chat_template = "llama-3.1"
    if "mistral" in args.model_name.lower():
        chat_template = "mistral"
    elif "gemma" in args.model_name.lower():
        chat_template = "gemma"
    print("Using chat template:", chat_template)
    model, tokenizer = get_generation_pipeline(args.model_name, chat_template=chat_template)
    kp_outputs = llm_inference(model, tokenizer, test_dataset, args.max_words, args.task)
    save_outputs(kp_outputs, test_dataset, args.output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract key points using Unsloth LLM.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSON.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for inference.")
    parser.add_argument("--max_kps", type=int, default=10, help="Maximum number of key points.")
    parser.add_argument("--max_words", type=int, default=15, help="Maximum words per key point.")
    parser.add_argument("--train", type=bool, default=False, help="Whether to train the model.")
    parser.add_argument("--task", type=str, default="cmv", help="Task type (default: cmv).")
    args = parser.parse_args()
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main(args)

