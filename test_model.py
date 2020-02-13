# Copyright 2020 Scott Liu

import os
import tensorflow as tf
import run_classifier
from train_model import delete_folder_contents
import flag_definitions

flags = tf.flags
FLAGS = flags.FLAGS


def do_testing(gpu=0):
    """
    This function evaluates our fine-tuning of the existing BERT model

    @param gpu: the index of the gpu you wish to use (default 0)
    @param prediction_mode: "regression" or "classification"
    """
    # expected environment variables
    os.environ["BERT_BASE_DIR"] = "pretrained/cased_L-12_H-768_A-12"
    os.environ["DATA_DIR"] = "dataset"
    os.environ["OUTPUT_DIR"] = "output"
    assert os.environ.get("BERT_BASE_DIR") is not None
    assert os.environ.get("DATA_DIR") is not None
    assert os.environ.get("OUTPUT_DIR") is not None

    # set the gpu index
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # set the required flags
    FLAGS.task_name = "topic"
    FLAGS.do_predict = True
    FLAGS.data_dir = os.environ.get("DATA_DIR")
    FLAGS.vocab_file = os.path.join(os.environ.get("BERT_BASE_DIR"), "vocab.txt")
    FLAGS.bert_config_file = os.path.join(os.environ.get("BERT_BASE_DIR"), "bert_config.json")
    FLAGS.init_checkpoint = os.path.join(os.environ.get("BERT_BASE_DIR"), "bert_model.ckpt")
    FLAGS.do_lower_case = False
    FLAGS.max_seq_length = 128
    FLAGS.output_dir = os.environ.get("OUTPUT_DIR")

    run_classifier.main(0)


def main():
    do_testing(gpu=0)
    #do_testing(gpu=0, prediction_mode="classification")


if __name__ == "__main__":
    main()
