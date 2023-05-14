import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering
import evaluate
from tqdm.auto import tqdm
import torch.cuda.profiler as profiler
import collections



# SETUP
max_length = 384
stride = 128
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
metric = evaluate.load("squad")
n_best = 20
max_answer_length = 30

##

def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def get_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)



def main():

    squad = load_dataset('squad')

    tokenized_dataset = squad.map(preprocess_training_examples, batched=True, remove_columns=squad["train"].column_names)

    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["validation"]

    testValidation = squad["validation"].map(preprocess_validation_examples, batched=True, remove_columns=squad["validation"].column_names)


    # train_dataset = squad["train"].map(preprocess_training_examples, batched=True, remove_columns=squad["train"].column_names)
    # validation_dataset = squad["validation"].map(preprocess_validation_examples, batched=True, remove_columns=squad["validation"].column_names)



    def compute_metrics(evalpred):
        predictions, label_ids = evalpred
        start_logits, end_logits = predictions
        return get_metrics(start_logits, end_logits, testValidation, squad["validation"])


    training_args = TrainingArguments(
                        output_dir='./experiments/', 
                        evaluation_strategy="steps",
                        eval_steps=1000, 
                        logging_steps=1000,
                        learning_rate=5e-5,
                        per_device_train_batch_size=8, #this is actually normal for LLM
                        num_train_epochs=0.05)

    model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=default_data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    profiler.start()
    trainer.train()
    profiler.stop()

if __name__ == '__main__':
    main()
