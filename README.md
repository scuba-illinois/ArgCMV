## ArgCMV: An Argument Summarization Benchmark for the LLM-era (EMNLP Main 2025)
#### Authors: Omkar Gurjar, Agam Goyal, Eshwar Chandrasekharan

This repository contains the code and data for the paper "ArgCMV: An Argument Summarization Benchmark for the LLM-era" which was accepted at EMNLP 2025.

### Data

We release the anonymized versions of the train, dev, and test splits of our dataset. Our entire dataset has been curated using **publicly available Reddit** posts from r/ChangeMyView subreddit. In order to ensure compliance with Reddit's data collection policy, we have only included the post/comment ids in our data. For any future research, practioners can use the ids to collect the actual comments using Reddit's API. 

The data is contained in the `data` folder as `json` files, where each item has the following keys:

- `post_id`: The id for the post. Starts with `t3_` 
- `comment_id`: The id for the comment. Starts with `t1_` for reply comments and `t3_` for original posts. 
- `topic_id`: The topic for the post. This is same as the `post_id` since both the topic and the OP argument are part of the original post. The topic always starts with `CMV:` and can be easily extracted using basic string processing.
- `argument_stance`:  `1` means pro topic, `-1` means against the topic.
- `topic_category`: The broad category for the topic of the post (LLM labeled).
- `kp`: the matching key points for the argument.
- `kps_mismatch`: key points which share the same topic, stance but aren't matched to the argument.
- `author_id`: an anonymized id for the author. Can be used to identify comments by the same author.


### Code

**Dependencies**: 
- `unsloth==2024.12.4` (https://docs.unsloth.ai/)
- `openai==1.78.0`
- `torch==2.5.1`

### Citation
If you use our code and/or data in your research, please cite as follows:
```
@inproceedings{
gurjar2025argcmv,
title={Arg{CMV}: An Argument Summarization Benchmark for the {LLM}-era},
author={Omkar Gurjar and Agam Goyal and Eshwar Chandrasekharan},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=eQeOPZCXBu}
}
```
