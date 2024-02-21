from transformers import AutoTokenizer, BertModel, BertTokenizer, RobertaModel, RobertaTokenizerFast
from groundingdino.util import box_ops, get_tokenlizer
import os


def self_get_tokenlizer(text_encoder_type):
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    print("final text_encoder_type: {}".format(text_encoder_type))

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_type, local_files_only=True)
    return tokenizer


get_tokenlizer.get_tokenlizer = self_get_tokenlizer
print("hijack get_tokenlizer success.")


def self_get_pretrained_language_model(text_encoder_type):
    if text_encoder_type == "bert-base-uncased":
        return BertModel.from_pretrained(text_encoder_type, local_files_only=True)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)
    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))


get_tokenlizer.get_pretrained_language_model = self_get_pretrained_language_model
print("hijack get_pretrained_language_model success.")
