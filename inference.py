import random
import argparse
import os
import itertools
import logging
from tqdm import tqdm
from fastchat.model import load_model, get_conversation_template, add_model_args
from app.prompt.template import complete_template, complete_template_todo
from app.static_analyzer.class_compose_tool import get_todo_methods, replace_method, retain_todo_method, retain_todo_method_with_ref
from app.util.io import extract_code, stream_jsonl, write_jsonl
from langchain_openai.chat_models import ChatOpenAI
import torch
import json
import ast

def inference(args):
    TASKID_METHODSIGNATURE = json.load(open("./constants/taskid_methodsignature.json"))
    TASKID_METHODDESCRIPTION = json.load(open("./constants/taskid_methoddescription.json"))
    is_openai = args.model_path.startswith("gpt")
    if is_openai:
        model = ChatOpenAI(model=args.model_path, temperature=args.temperature)
    else:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        model, tokenizer = load_model(
            args.model_path,
            device=args.device,
            num_gpus=args.num_gpus,
            max_gpu_memory=args.max_gpu_memory,
            load_8bit=args.load_8bit,
            cpu_offloading=args.cpu_offloading,
            revision=args.revision,
            debug=args.debug,
        )
        tokenizer.pad_token = tokenizer.eos_token
        model = model.to(torch.bfloat16)

    # def query(code, code_context, todo_method_name=None):
    def query(code, code_context, target_signature=None):
        lc_messages = complete_template_todo.format_messages(
            code_context=code_context,
            code=code,
        )

        if is_openai:
            prompt = lc_messages[0].content + "\n" + lc_messages[1].content
            outputs = model.invoke(lc_messages).content
            log_probabilities = [1.0]
            outputs_decoded_by_token = [outputs[:10], outputs[10:]] # totally artificial, for consistency with non-openai models
            return prompt, outputs_decoded_by_token, log_probabilities, -1
        else:
            conv = get_conversation_template(args.model_path)
            if "{system_message}" in conv.system_template:
                conv.system_message = lc_messages[0].content
            else:
                conv.append_message(conv.roles[0], lc_messages[0].content)
            conv.append_message(conv.roles[0], lc_messages[1].content)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            prompt = prompt.replace("""
### Human: Got any creative ideas for a 10 year old\u2019s birthday?
### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!
""", "")

            if target_signature:
                prompt = prompt.replace("<METHOD_SIGNATURE>", f"`{target_signature}`")
            
            try:
                # Run inference
                inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
                prompt_len = inputs["input_ids"].shape[1]
                gen_outputs = model.generate(
                    **inputs,
                    do_sample=True if args.temperature > 1e-5 else False,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_scores=True  
                )
                output_ids = gen_outputs.sequences
                generated_ids = output_ids[0][prompt_len:]

                log_probabilities = []
                for t, scores_t in enumerate(gen_outputs.scores):
                    logprobs_t = scores_t.log_softmax(dim=-1)
                    token_id = generated_ids[t].unsqueeze(0)
                    token_lp = logprobs_t.gather(1, token_id.unsqueeze(-1)).squeeze()
                    log_probabilities.append(token_lp.item())

                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
                
                # outputs = tokenizer.decode(
                #     output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                # )
                outputs_decoded_by_token = [tokenizer.decode([tid], skip_special_tokens=True, spaces_between_special_tokens=False) for tid in output_ids]
            
            except torch.OutOfMemoryError:
                outputs_decoded_by_token = ["FAILED", "_", "OOM"]
                log_probabilities = []

        return prompt, outputs_decoded_by_token, log_probabilities, prompt_len

    output_path = args.output.split(".jsonl")[0] + f"-{os.getpid()}.jsonl"
    data_path = os.path.join('datasets', f'{args.context}-context', f'data-{args.project}-REF.jsonl')
    tasks = list(stream_jsonl(data_path))
    samples = list(stream_jsonl(output_path)) if os.path.exists(output_path) else []

    ids_to_keep = None
    if args.all_ids_dict != "all":
        ids_to_keep = ast.literal_eval(args.all_ids_dict)["JavaBench"][args.split]    

    for task, sample_idx in tqdm(itertools.islice(itertools.product(tasks, range(args.num_sample)), len(samples), None), total=len(tasks) * args.num_sample, initial=len(samples)):
        
        if args.mode == "holistic":
            solution_idx = "Java--JavaBench--TaskID::{0}--GeneratedBy::{1}--SampleID::{2}".\
                                    format(task["task_id"].split('.java')[0], args.model_path.split("/")[-1], sample_idx)
            prompt, outputs, log_probabilities = query(task["code"], task["code_context"])
            outputs = ''.join(outputs)
            samples.append(dict(
                task_id=task["task_id"],
                sample_idx=sample_idx,
                generated_by = args.model_path.split("/")[-1],
                target=task["target"],
                prompt=prompt,
                completion=outputs,
                log_probabilities=log_probabilities,
                solution_idx=solution_idx,
            ))
        elif args.mode == "independent":
            result = task["code"]
            mediate = []

            todo_methods = get_todo_methods(result)
            progress = tqdm(todo_methods)
            for todo_method in progress:
                progress.set_description(f"{todo_method['name']} {todo_method['seq']}")
                source = retain_todo_method(task["code"], todo_method["name"], todo_method["seq"])
                prompt, outputs, log_probabilities = query(source, task["code_context"])
                outputs = ''.join(outputs)
                result, _ = replace_method(result, extract_code(outputs), todo_method["name"], todo_method["seq"])
                new_mediate = dict(
                    name=todo_method["name"],
                    seq=todo_method["seq"],
                    prompt=prompt,
                    completion=outputs,
                    log_probabilities=log_probabilities,
                )
                mediate.append(new_mediate)
                logging.info(f"{todo_method['name']} {todo_method['seq']} {new_mediate}")

            samples.append(dict(
                task_id=task["task_id"],
                sample_idx=sample_idx,
                generated_by = args.model_path.split("/")[-1],
                target=task["target"],
                completion=result,
                mediate=mediate,
                solution_idx=solution_idx,
            ))
        elif args.mode == "independent-with-ref":
            # in this mode, the idea is to provide the model with the whole class implemented as the reference solution
            # but only one method is left as TODO in the source code to be completed by the model
            result = task["code"]
            mediate = []

            todo_methods = get_todo_methods(result)
            progress = tqdm(todo_methods)
            for todo_method in progress:

                ##########
                if f"{task['task_id'].split('.java')[0]}:{todo_method['name']}:{todo_method['seq']}" not in ids_to_keep and ids_to_keep is not None:
                    continue
                ##########

                task_idx = f"{task['task_id'].split('.java')[0]}:{todo_method['name']}:{todo_method['seq']}"
                solution_idx = "Java--JavaBench--TaskID::{0}--GeneratedBy::{1}--SampleID::{2:02d}".\
                                    format(task_idx, args.model_path.split("/")[-1], sample_idx)
                progress.set_description(f"{todo_method['name']} {todo_method['seq']}")

                # read the reference source code
                with open(os.path.join('projects', f'{args.project}-Solution', 'src', 'main', 'java', task['target']), 'r') as r:
                    source_ref = r.read()
                
                # substitute all other methods with the reference implementation and keep the TODO one
                prompt = retain_todo_method_with_ref(source=task['code'], source_ref=source_ref, todo_method=todo_method)
                # prompt, outputs_decoded_by_token, log_probabilities, prompt_len = query(prompt, task["code_context"], todo_method['name'])
                target_signature = TASKID_METHODSIGNATURE[task_idx]
                target_description = TASKID_METHODDESCRIPTION[task_idx]
                prompt, outputs_decoded_by_token, log_probabilities, prompt_len = query(prompt, task["code_context"], target_signature)
                
                outputs = ''.join(outputs_decoded_by_token)
                # paste the generated patch into the reference class implementation
                # result, patch = replace_method(prompt, extract_code(outputs), todo_method["name"], todo_method["seq"])

                if extract_code(outputs): # this is the case in which the completion of the model contains a ```java ... ``` block
                    result, patch = replace_method(prompt, extract_code(outputs), todo_method["name"], todo_method["seq"])
                else:
                    result, patch = "EXTRACTION_FAILED", ""

                
                # extract the log probabilities corresponding to the generated patch only
                if patch and len(patch.split('\n')) > 1 and not is_openai:
                    patch_wo_signature = '\n'.join(patch.split('\n')[1:])
                    before_patch = outputs.split(patch_wo_signature)[0]
                    num_tokens_upto_patch = len(tokenizer.encode(before_patch, add_special_tokens=False))
                    number_of_patch_tokens = len(tokenizer.encode(patch_wo_signature, add_special_tokens=False))

                    # assert patch_wo_signature.strip() == \
                    # ''.join(outputs_decoded_by_token[num_tokens_upto_patch : num_tokens_upto_patch + number_of_patch_tokens]).strip()

                    if num_tokens_upto_patch > 0 and num_tokens_upto_patch + number_of_patch_tokens <= len(log_probabilities):
                        log_probabilities = log_probabilities[num_tokens_upto_patch : num_tokens_upto_patch + number_of_patch_tokens]

                samples.append(dict(
                    # prompt_len=prompt_len,
                    task_idx=task_idx,
                    sample_idx=sample_idx,
                    func_name=todo_method["name"],
                    signature=target_signature,
                    description=target_description,
                    func_seq=todo_method["seq"],
                    generated_by = args.model_path.split("/")[-1],
                    target=task["target"],
                    prompt=prompt,
                    method=patch,
                    completion=result,
                    raw_output=outputs,
                    log_probabilities=log_probabilities,
                    solution_idx=solution_idx,
                    task_id=task["task_id"],
                ))
        elif args.mode == "incremental":
            result = task["code"]
            mediate = []

            todo_methods = get_todo_methods(result)
            if args.incremental_mode == "rev":
                todo_methods = reversed(todo_methods)
            elif args.incremental_mode == "rand":
                random.shuffle(todo_methods)
            progress = tqdm(todo_methods)
            for todo_method in progress:
                progress.set_description(f"{todo_method['name']} {todo_method['seq']}")
                source = retain_todo_method(result, todo_method["name"], todo_method["seq"])
                prompt, outputs, log_probabilities = query(source, task["code_context"])
                outputs = ''.join(outputs)
                result, _ = replace_method(result, extract_code(outputs), todo_method["name"], todo_method["seq"])
                new_mediate = dict(
                    name=todo_method["name"],
                    seq=todo_method["seq"],
                    prompt=prompt,
                    completion=outputs,
                    log_probabilities=log_probabilities,
                )
                mediate.append(new_mediate)
                logging.info(f"{todo_method['name']} {todo_method['seq']} {new_mediate}")

            samples.append(dict(
                task_id=task["task_id"],
                target=task["target"],
                completion=result,
                mediate=mediate,
            ))
        if os.path.dirname(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        write_jsonl(output_path, samples)


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(format='%(levelname)s %(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S', filename=f"logs/inference-{os.getpid()}.log", filemode="w", level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--mode", 
        type=str,
        choices=["holistic", "independent", "independent-with-ref", "incremental"],
        default="holistic",
    )

    parser.add_argument("--project", type=str, required=True, choices=['PA19', 'PA20', 'PA21', 'PA22'])
    parser.add_argument("--context", type=str, required=True, choices=['maximum', 'minimum', 'selective'])
    parser.add_argument("--all_ids_dict", default=None, type=str)
    parser.add_argument("--split", choices=["train", "test", "val", "all"], type=str, help="Subset of the data to run on (train/val/test/all).")
    parser.add_argument("--num-sample", type=int, default=10)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--incremental-mode", type=str, choices=["seq", "rev", "rand"], default="seq")

    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    inference(args)
