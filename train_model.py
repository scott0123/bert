# Copyright 2020 Scott Liu

import os
import shutil
import tensorflow as tf
import run_classifier
import flag_definitions

flags = tf.flags
FLAGS = flags.FLAGS


def delete_folder_contents(path):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)


def do_training(gpu=0, prediction_mode="regression", restart=True):
    """
    This function performs fine-tuning on the existing BERT model

    @param gpu: the index of the gpu you wish to use (default 0)
    @param prediction_mode: "regression" or "classification"
    @param restart: should we delete everything in ./output and start from scratch
    """
    os.environ["BERT_BASE_DIR"] = "pretrained/cased_L-12_H-768_A-12"
    os.environ["DATA_DIR"] = "dataset"
    os.environ["OUTPUT_DIR"] = "output"
    # expected environment variables
    assert os.environ.get("BERT_BASE_DIR") is not None
    assert os.environ.get("DATA_DIR") is not None
    assert os.environ.get("OUTPUT_DIR") is not None

    # set the gpu index
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # set the required flags
    FLAGS.task_name = "topic"
    FLAGS.do_train = True
    FLAGS.do_eval = True
    FLAGS.data_dir = os.environ.get("DATA_DIR")
    FLAGS.vocab_file = os.path.join(os.environ.get("BERT_BASE_DIR"), "vocab.txt")
    FLAGS.bert_config_file = os.path.join(os.environ.get("BERT_BASE_DIR"), "bert_config.json")
    FLAGS.init_checkpoint = os.path.join(os.environ.get("BERT_BASE_DIR"), "bert_model.ckpt")
    FLAGS.do_lower_case = False
    FLAGS.max_seq_length = 128
    FLAGS.train_batch_size = 1
    FLAGS.learning_rate = 2e-5
    FLAGS.num_train_epochs = 1
    FLAGS.output_dir = os.environ.get("OUTPUT_DIR")

    if restart:
        delete_folder_contents(FLAGS.output_dir)
        open(".gitkeep", "w").close()
    run_classifier.main(0)


def main():
    do_training(gpu=0)


if __name__ == "__main__":
    main()
