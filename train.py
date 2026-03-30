import json
import os
import re
import string
import numpy as np
import torch
import torchaudio
import argparse
from datasets import load_dataset
import evaluate
from transformers import (Trainer, TrainingArguments, Wav2Vec2CTCTokenizer,
                          Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC,
                          Wav2Vec2Processor)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for Apple MPS if needed

DATASET_DIR = 'data/cv-corpus-25.0-2026-03-09/yue'
CLIPS_DIR = f"{DATASET_DIR}/clips"
OUT_MODEL = 'wav2vec2-xls-r-1b-cantonese-yue'
OUT_MODEL_DIR = f"./{OUT_MODEL}"

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
                    default="facebook/wav2vec2-xls-r-1b")
parser.add_argument('--unfreeze', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--warmup', type=int, default=500)
args = parser.parse_args()
print(f"args: {args}")

# ---------------------------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------------------------
full_common_voice_train = load_dataset(
    "csv",
    data_files=[f'{DATASET_DIR}/train.tsv', f'{DATASET_DIR}/validated.tsv'],
    delimiter="\t",
    split="train")
# common_voice_train = full_common_voice_train.shuffle(
#     seed=1997).select(range(100))
common_voice_train = full_common_voice_train
common_voice_test = load_dataset(
    'csv',
    data_files=f'{DATASET_DIR}/test.tsv',
    delimiter="\t",
    split="train[:10%]"
)

# Remove unused columns - filter to only columns that actually exist
unused_cols = ["accents", "age", "client_id", "down_votes", "gender", "locale",
               "segment", "up_votes"]
existing_train_cols = [c for c in unused_cols
                       if c in common_voice_train.column_names]
existing_test_cols = [c for c in unused_cols
                      if c in common_voice_test.column_names]
common_voice_train = common_voice_train.remove_columns(existing_train_cols)
common_voice_test = common_voice_test.remove_columns(existing_test_cols)

# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
chars_to_ignore_regex = (r'[\丶\,\?\.\!\-\;\:"\"\%\'\"\�\．\⋯\！\－\：\–\。'
                         r'\》\,\）\,\？\；\～\~\…\︰\，\（\」\‧\《\﹔\、\—'
                         r'\／\,\「\﹖\·\']')


def remove_special_characters(batch):
    sen = re.sub(chars_to_ignore_regex, '', batch["sentence"]).lower() + " "
    if "d" in sen:
        if len([c for c in sen if c in string.ascii_lowercase]) == 1:
            sen = sen.replace("d", "啲")
    batch["sentence"] = sen
    return batch


common_voice_train = common_voice_train.map(remove_special_characters)
common_voice_test = common_voice_test.map(remove_special_characters)


# ---------------------------------------------------------------------------
# 3. Build vocabulary
# ---------------------------------------------------------------------------
def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = common_voice_train.map(
    extract_all_chars, batched=True,
    batch_size=64,
    remove_columns=common_voice_train.column_names,)
vocab_test = common_voice_test.map(
    extract_all_chars, batched=True,
    batch_size=64,
    remove_columns=common_voice_test.column_names,)

vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
# remove english/ascii chars from vocab_list so tokenizer maps them to [UNK]
vocab_list = [char for char in vocab_list if not char.isascii()]
vocab_list.append(" ")  # re-add space

vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

with open("vocab.json", "w", encoding="utf-8") as vocab_file:
    json.dump(vocab_dict, vocab_file, ensure_ascii=False)

# ---------------------------------------------------------------------------
# 4. Create processor
# ---------------------------------------------------------------------------
tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]",
                                 pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                             sampling_rate=16000,
                                             padding_value=0.0,
                                             do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor,
                              tokenizer=tokenizer)
processor.save_pretrained(OUT_MODEL_DIR)

# ---------------------------------------------------------------------------
# 5. Load and resample audio
# ---------------------------------------------------------------------------
resamplers = {
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
    22050: torchaudio.transforms.Resample(22050, 16000),
}


def load_and_resample(batch):
    filepath = os.path.join(CLIPS_DIR, batch["path"])
    speech_array, sampling_rate = torchaudio.load(filepath)
    if sampling_rate != 16000:
        batch["speech"] = resamplers[sampling_rate](
            speech_array).squeeze().numpy()
    else:
        batch["speech"] = speech_array.squeeze().numpy()
    batch["sampling_rate"] = 16_000
    batch["target_text"] = batch["sentence"]
    return batch


common_voice_train = common_voice_train.map(
    load_and_resample,
    remove_columns=common_voice_train.column_names,)
common_voice_test = common_voice_test.map(
    load_and_resample,
    remove_columns=common_voice_test.column_names,)


# ---------------------------------------------------------------------------
# 6. Prepare features and labels
# ---------------------------------------------------------------------------
def prepare_dataset(batch):
    batch["input_values"] = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"][0]).input_values
    batch["labels"] = processor.tokenizer(batch["target_text"]).input_ids
    return batch


common_voice_train = common_voice_train.map(
    prepare_dataset,
    remove_columns=common_voice_train.column_names,
    batch_size=16,
    batched=True,
)
common_voice_test = common_voice_test.map(
    prepare_dataset,
    remove_columns=common_voice_test.column_names,
    batch_size=16,
    batched=True,
)


# ---------------------------------------------------------------------------
# 7. Data collator
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or
                :class:`~transformers.tokenization_utils_base.PaddingStrategy`,
                `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to
                the model's padding side and padding index) among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence
                in the batch (or no padding if only a single sequence if
                provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the
                argument :obj:`max_length` or to the maximum acceptable input
                length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e.,
                can output a batch with sequences of different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and
            optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally
            padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on
            NVIDIA hardware with compute capability >= 7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# ---------------------------------------------------------------------------
# 8. Metrics
# ---------------------------------------------------------------------------
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}


# ---------------------------------------------------------------------------
# 9. Load model
# ---------------------------------------------------------------------------
model = Wav2Vec2ForCTC.from_pretrained(
    args.model,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
)

if not args.unfreeze:
    model.freeze_feature_encoder()

# ---------------------------------------------------------------------------
# 10. Training
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUT_MODEL_DIR,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=400,
    num_train_epochs=40,
    fp16=torch.cuda.is_available(),  # only enable on CUDA
    logging_strategy="steps",
    logging_steps=400,
    learning_rate=args.lr,
    warmup_steps=args.warmup,
    save_steps=2376,  # every 3 epoch with batch_size 8
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_test,
    processing_class=processor.feature_extractor,
)

trainer.train(resume_from_checkpoint=True)
