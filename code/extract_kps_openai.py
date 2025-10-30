import torch
from tqdm import tqdm
import json
import re
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, Timeout, APIConnectionError
from prompts import system_prompt_kp_cmv, base_prompt_kp_cmv

@retry(
    wait=wait_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((RateLimitError, APIError, Timeout, APIConnectionError)),
    reraise=True
)
def generate_kps_from_model(client, model_name, prompt):
    """Generate key points from OpenAI model using chat completion."""
    response = client.chat.completions.create(
        model=model_name,
        messages=prompt,
        temperature=0
    )
    return response.choices[0].message.content


def llm_inference(model_name, openai_key, dataset, max_kps, max_words):
    """Run LLM inference to extract key points from clusters in the dataset using OpenAI API."""
    final_outputs = []
    client = OpenAI(api_key=openai_key)
    num_topics = 500
    i = 0
    for topic, topic_data in tqdm(dataset.items()):
        for stance, stance_data in topic_data.items():
            print(f"Topic: {topic}, Stance: {stance}, Clusters: {len(stance_data)}")
            clusters = ""
            for cluter_id, cluster_data in stance_data.items():
                clusters += f"**Cluster {cluter_id}** \n- " + "\n- ".join([item["argument"] for item in cluster_data]) + "\n\n"
            system_prompt_text = system_prompt_kp_cmv.format(kp_token_length=max_words)
            text_stance = "supporting" if stance == "1" else "opposing"
            prompt_text = base_prompt_kp_cmv.format(
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
            try:
                output_text = generate_kps_from_model(client, model_name, messages)
            except Exception as e:
                print(f"[Error] Failed for topic: '{topic}', error: {str(e)}")
                output_text = "[Error] API call failed after retries"
            row = {
                "topic": topic,
                "stance": stance,
                "llm_output": output_text,
                "num_users": len(stance_data),
                "num_arguments": sum(len(cluster_data) for cluster_data in stance_data.values())
            }
            final_outputs.append(row)
        if i >= num_topics:
            break
        i += 1
    return final_outputs


def load_data(data_path, by_adu=False):
    """Load dataset from a JSON file."""
    with open(data_path, "r") as f:
        data = json.load(f)
    return data

def save_outputs(kp_outputs, output_path):
    """Save extracted key points and metadata to a JSON file."""
    final_output = []
    for item in kp_outputs:
        topic = item["topic"]
        stance = item["stance"]
        kps = re.findall(r"- (.*)", item["llm_output"])
        kps = [arg.strip() for arg in kps]
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

def main(args):
    """Main function to run key point extraction pipeline using OpenAI."""
    with open('/u/ogurjar/openai_key') as f:
        openai_key = f.read().strip()
    test_path = args.file_path
    test_dataset = load_data(test_path)
    model_name = args.model_name
    print(f"Model name: {model_name}")
    kp_outputs = llm_inference(model_name, openai_key, test_dataset, args.max_kps, args.max_words)
    save_outputs(kp_outputs, args.output_path)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Extract key points using OpenAI chat completion.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to input data file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSON.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name for inference.")
    parser.add_argument("--max_kps", type=int, default=10, help="Maximum number of key points.")
    parser.add_argument("--max_words", type=int, default=15, help="Maximum words per key point.")
    parser.add_argument("--train", type=bool, default=False, help="Whether to train the model.")

    args = parser.parse_args()
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    main(args)

   