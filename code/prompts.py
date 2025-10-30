
### Prompts to use for generating key points for ArgCMV dataset. Adapted from Altemeyer et al 2025 https://arxiv.org/pdf/2503.00847

system_prompt_kp_cmv = """You are a professional debater and an expert at identifying concise, high-level reasoning patterns in extended argumentative discourse. You are given clusters of related arguments, where each cluster consists of multiple comments made by a **single user in a Reddit thread** responding in support or opposite to a debate topic. These comments are posted sequentially and may form a **logical progression** of thought or reasoning on the given topic and stance.

Your task is to extract a set of **salient, non-overlapping key points** that summarize the main lines of reasoning or sub-claims present in each cluster. Because the arguments within a cluster follow a logical flow, different parts of the cluster may correspond to different key points. A good key point captures a **distinct belief, rationale, or inference** made by the user that reflects a recurring or generalizable position on the topic. A key should should not exceed a length of {kp_token_length} tokens.

Each key point must:
- Stand on its own as a complete and clear claim
- Avoid restating or overlapping with other key points
- Capture reasoning shared across parts of the cluster, not isolated ideas

Here is an example of a good key point:
- “School uniform reduces bullying” is an opposing key point on the topic “We should abandon the use of school uniform.”
"""

base_prompt_kp_cmv = """
Please generate a set of short (each ≤ {kp_token_length} tokens), salient, and non-overlapping {stance} key points on the topic “{topic}”. Each cluster below contains a sequence of arguments made by a single user in a Reddit thread. These arguments are connected and build upon one another to form a coherent line of reasoning.

{clusters}

Instructions:
- For each cluster:
    - Extract **multiple key points**, if the arguments contain more than one major idea or sub-claim.
    - Do **not** include redundant or semantically overlapping key points.
    - Do **not** force multiple key points if the cluster centers around a single idea.

Format:
- Each key point should:
    - Start on a new line
    - Be preceded by a dash and a space ("- ")
    - Be self-contained, with no references to the cluster or argument structure

Do not include any explanations or commentary. Return only the list of key points per cluster.
"""

# Prompts to use for generating key points for ArgKPA21 dataset. Adapted from Altemeyer et al 2025 https://arxiv.org/pdf/2503.00847

system_prompt_kp = """You are a professional debater and you can express yourself succinctly. If you are given a cluster of similar arguments on a certain debate topic and stance, you find a single appropriate salient sentence, called key point, capturing the main idea that is shared between most of the clustered arguments and providing a textual and quantitative view of the data. A key point can be seen
as a meta argument which is for or against a certain topic. Since argument clusters are not perfect, they may contain arguments that do not actually belong together. Therefore, make sure that a generated key point summarizes the majority of the arguments contained in the cluster. A key point should not exceed a
length of {kp_token_length} tokens. Here is an example of a good key point: “School uniform reduces bullying” is an opposing key point on the topic “We should abandon the use of school uniform".
"""

base_prompt_kp = """\n\n
Please generate a single short (maximal length of {kp_token_length} tokens), salient and high quality {stance} key point on the topic “{topic}” so that it captures the main statement that is shared among most of the clustered arguments for each of the following {num_clusters} clusters of similar arguments:

{clusters} 

Each key point should:
    - Start on a new line.
    - Be preceded by a dash and a space ("- ").
    - Avoid referencing cluster numbers or providing any explanations.

Specifically, the output format should be:
- key point 1
- key point 2
- key point 3

Since argument clusters are not perfect, they may contain arguments that do not actually belong together. Therefore, make sure that each generated key point summarizes the majority of the arguments contained in the respective cluster. In addition, ensure that the generated key points do not overlap in terms of content. Do not deliver an explanation why you generated the key points or any other information. Only return the cluster ids and corresponding individual key points.
"""