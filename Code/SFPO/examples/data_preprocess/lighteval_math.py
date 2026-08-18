# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the math dataset to parquet format
"""

import os
import datasets

import argparse



def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    open_braces = 0
    while i < len(string):
        if string[i] == "{":
            open_braces += 1
        elif string[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                return string[idx:i + 1]
        i += 1
    return None


def remove_boxed(s):
    if "\\boxed " in s:
        return s[len("\\boxed "):]
    return s[len("\\boxed{"):-1]


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def make_prefix(dp, template_type):
    problem = dp['problem']
    # NOTE: also need to change reward_score/countdown.py
    if template_type == 'base':
        """This works for any base model"""
        prefix = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please solve the following math problem: {problem}. Show your steps in <think> </think> tags. And return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Assistant: Let me solve this step by step <think>"""
    elif template_type == 'qwen-instruct':
        """This works for Qwen Instruct Models"""
        prefix = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Please solve the following math problem: {problem}. Show your steps in <think> </think> tags. And return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""

    prefix = f"""Please solve the following math problem: {problem}. The assistant first thinks about the reasoning process step by step and then provides the user with the answer. Return the final answer in \\boxed{{}} tags, for example \\boxed{{1}}. Let's solve this step by step. """
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/lighteval-math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', type=int, default=7500)
    parser.add_argument('--test_size', type=int, default=5000)
    parser.add_argument('--train_only', action='store_true',
                        help='Only write the selected train split; skip test preprocessing.')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'xDAN2099/lighteval-MATH'
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train'].select(range(TRAIN_SIZE))
    test_dataset = None if args.train_only else dataset['test'].select(range(TEST_SIZE))

    # instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            # question = example.pop('problem')

            # question = question + ' ' + instruction_following
            question = make_prefix(example, template_type=args.template_type)

            answer = example.pop('solution')
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    if test_dataset is not None:
        test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    if test_dataset is not None:
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # print data source and length
    print(f"Data source: {data_source}")
    print(f"Length of train dataset: {len(train_dataset)}")

    if hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
