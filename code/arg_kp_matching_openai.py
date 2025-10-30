import torch
from tqdm import tqdm
import pandas as pd
import json
import re
from prompts import *
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, Timeout, APIConnectionError
from pprint import pprint

system_prompt="""You are an expert debater and a professional at identifying concise, high-level, salient sentences called key points given an argument. You will be given argument related to a topic and stance. Additionally, you will be given a set of key points which were created by human experts for this topic and stance. Your task is to identify the key points that are present in the argument. A key point is considered present in the argument if the main idea is expressed clearly, even if reworded. Your output should be a **Python-style list** of the **indices of matching KPs**, e.g., `[0, 2]`.
For example, the following argument and key points are given for the topic "We should ban the use of child actors" and opposing stance:

Argument: Banning child actors would ignore the fact that many children genuinely enjoy acting and choose to pursue it as a career. With appropriate regulation and adult supervision, they can work in safe environments where their well-being is prioritized. Moreover, child acting can offer early exposure to professional opportunities that build confidence, discipline, and creative skills. Instead of banning, we should focus on enforcing strict industry protections to prevent exploitation.

Key Points:
0 Child performers should not be banned as long as there is supervision/regulation.
1 Acting helps children build confidence and public speaking skills.
2 Child acting provides families with income opportunities.
3 Child actors have the right to choose their career.

Output: [0, 1, 3]
"""


user_prompt="""Given the argument and corresponding key points {stance} the topic "{topic}", identify the key points that are present in the argument. Carefully analyze each key point one by one and check if its contained in the argument. Your output should be a **Python-style list** of the **indices of matching KPs**, e.g., `[0, 2]`. Only output the list of matching KPs. Do not include any other text or explanation.
Argument: {argument}
Key Points:
{kps}
Output: """

# Prompts to use for matching key points to arguments for ArgKP dataset. Adapted from Altemeyer et al 2025 https://arxiv.org/pdf/2503.00847

system_prompt_argkp = """You are an expert debater and a professional at identifying concise, high-level, salient sentences called key points given an argument. You will be given argument related to a topic and stance. Additionally, you will be given a key point which were created by human experts for this topic and stance. Your task is to identify if the given key point is present in the argument. A key point is considered present in the argument if the main idea is expressed clearly, even if reworded. Your output should be a single word, either "yes" or "no", indicating whether the key point is present in the argument.

For example, the following argument and key points are given for the topic "We should ban the use of child actors" and opposing stance:

Argument: child actors are not able to make their own decisions and are placed in situations that they do not understand for the financial benefit of others and should be banned.

Key Point: Child performers are at risk of exploitation
Output: yes

Key Point: Being a performer harms the child's education
Output: no
"""

user_prompt_argkp = """Given the argument and corresponding key point {stance} the topic "{topic}", identify if the key point is present in the argument. Carefully analyze the key point and check if its contained in the argument. Your output should be a single word, either "yes" or "no", indicating whether the key point is present in the argument.

Argument: {argument}
Key Point: {kp}
Output: """

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

def argkp_matching_inference(model_name, openai_key, arguments):
    """Run LLM inference to match arguments to key points for ArgKP dataset."""
    print("Using model:", model_name)
    final_outputs = []
    client = OpenAI(api_key=openai_key)
    predictions = []
    labels = []
    for i in tqdm(range(len(arguments))):
        argument = arguments[i]['argument']
        topic = arguments[i]['topic']
        stance = arguments[i]['stance']
        kp = arguments[i]['kp']
        label = arguments[i]['label']
        user_prompt_text = user_prompt_argkp.format(
            topic=topic,
            stance="supporting" if stance == "1" else "opposing",
            argument=argument,
            kp=kp
        )
        messages = [
            {"role": "system", "content": system_prompt_argkp},
            {"role": "user", "content": user_prompt_text}
        ]
        try:
            output_text = generate_kps_from_model(client, model_name, messages)
        except Exception as e:
            print(f"[Error] Failed for argument: '{argument}', error: {str(e)}")
            output_text = "[Error] API call failed after retries"
        row = {
            "topic": topic,
            "stance": stance,
            "argument": argument,
            "kp": kp,
            "llm_output": output_text,
            "label": label
        }
        final_outputs.append(row)
        pred = 1 if output_text.strip().lower() == "yes" else 0
        predictions.append(pred)
        labels.append(label)
    df = pd.DataFrame(final_outputs)
    df['predictions'] = predictions
    return df

def llm_inference(model_name, openai_key, argument_map, kps_map, output_path):
    """Perform LLM inference to match extracted KPs to arguments for ArgCMV dataset."""
    print("Using model:", model_name)
    final_outputs = []
    client = OpenAI(api_key=openai_key)
    num_topics = 3
    i = 0
    for topic, topic_data in tqdm(argument_map.items()):
        for stance, stance_data in topic_data.items():
            kps = kps_map.get(topic, {}).get(stance, [])
            formatted_kps = [f"{i} {kp}" for i, kp in enumerate(kps)]
            kps_text = "\n".join(formatted_kps)
            for item in stance_data:
                argument = item['argument']
                arg_id = item['arg_id']
                user_prompt_text = user_prompt.format(
                    topic=topic,
                    stance="supporting" if stance == "1" else "opposing",
                    argument=argument,
                    kps=kps_text
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_text}
                ]
                try:
                    output_text = generate_kps_from_model(client, model_name, messages)
                except Exception as e:
                    print(f"[Error] Failed for topic: '{topic}', error: {str(e)}")
                    output_text = "[Error] API call failed after retries"
                row = {
                    "topic": topic,
                    "stance": stance,
                    "arg_id": arg_id,
                    "argument": argument,
                    "llm_output": output_text,
                    "kps": kps,
                }
                final_outputs.append(row)
        if (i+1) % 100 == 0:
            save_outputs(final_outputs, output_path)
            print(f"Saved outputs after processing {i+1} topics.")
        i += 1
    return final_outputs

def read_data(arguments_path, kps_path):
    """Read arguments and key points from JSON files and organize them by topic and stance."""
    with open(arguments_path, 'r') as f:
        arguments = json.load(f)
    with open(kps_path, 'r') as f:
        kps = json.load(f)
    argument_map = {}
    for item in arguments:
        topic = item['topic']
        stance = item['arg_stance']
        argument = item['argument']
        arg_id = item['arg_id']
        if topic not in argument_map:
            argument_map[topic] = {}
        if stance not in argument_map[topic]:
            argument_map[topic][stance] = []
        argument_map[topic][stance].append({
            'argument': argument,
            'arg_id': arg_id
        })
    kps_map = {}
    for item in kps:
        topic = item['topic']
        stance = item['stance']
        kps_list = item['kps']
        if topic not in kps_map:
            kps_map[topic] = {}
        if stance not in kps_map[topic]:
            kps_map[topic][stance] = []
        kps_map[topic][stance] = kps_list
    return argument_map, kps_map

def save_outputs(final_outputs, output_path):
    """Parse and save matching key points to a JSON file."""
    for item in final_outputs:
        try:
            matching_kp_idx = re.findall(r'\[(.*?)\]', item["llm_output"])[0] if "[" in item["llm_output"] else item["llm_output"]
            matching_kp_idx = matching_kp_idx.strip()
            if len(matching_kp_idx) == 0:
                item["matching_kps"] = []
                continue
            matching_kp_idx = matching_kp_idx.split(",")
            matching_kp_idx = [int(kp.strip()) for kp in matching_kp_idx]
            kps = item["kps"]
            matching_kps = [kps[i] for i in matching_kp_idx]
            item["matching_kps"] = matching_kps
        except Exception as e:
            print(f"[Error] Failed to parse matching KPs for topic: '{item['topic']}', error: {str(e)}")
    with open(output_path, "w") as f:
        json.dump(final_outputs, f, indent=4)


def argkp_matching(model, openai_key):
    """Few-shot evaluation on the ArgKP test dataset."""
    with open("/u/ogurjar/causal_llm/Key-Point-Analysis/Datasets/ArgKPData/test.json") as f:
        arguments = json.load(f)
    data = []
    for item in arguments:
        topic = item['topic']
        stance = item['arg_stance']
        argument = item['argument']
        for x in item['kps']:
            data.append({
                "topic": topic,
                "stance": stance,
                "argument": argument,
                "kp": x['key_point'],
                "label": 1
            })
        for x in item["kps_mismatch"]:
            data.append({
                "topic": topic,
                "stance": stance,
                "argument": argument,
                "kp": x['key_point'],
                "label": 0
            })
    final_output = argkp_matching_inference(model, openai_key, data)
    from sklearn.metrics import classification_report
    print(classification_report(final_output['label'], final_output['predictions'], digits=4))

def main(args):
    """Main function to run argument-KP matching or ArgCMV KP extraction."""
    with open('/u/ogurjar/openai_key') as f:
        openai_key = f.read().strip()
    if args.task_name == "argcmv":
        argument_map, kps_map = read_data(args.arguments_path, args.kps_path)
        final_outputs = llm_inference(args.model_name, openai_key, argument_map, kps_map, args.output_path)
        save_outputs(final_outputs, args.output_path)
    elif args.task_name == "argkp":
        argkp_matching(args.model_name, openai_key)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the OpenAI API for KP extraction.")
    parser.add_argument("--task_name", type=str, required=True, choices=["argcmv", "argkp"], default="argcmv")
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model name.")
    parser.add_argument("--arguments_path", type=str, required=True, help="Path to the arguments JSON file.")
    parser.add_argument("--kps_path", type=str, required=True, help="Path to the KPs JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSON file.")

    args = parser.parse_args()
    main(args)


