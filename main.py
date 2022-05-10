from os import makedirs
from os.path import join
import logging
import numpy as np
import torch
import random
from args import define_main_parser

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from dataset.prsa import PRSADataset
from dataset.card import TransactionDataset
from models.modules import TabFormerBertLM, TabFormerGPT2
from misc.utils import random_split_dataset
from dataset.datacollator import TransDataCollatorForLanguageModeling
import h5py as h5
import json

logger = logging.getLogger(__name__)
log = logger
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def jiri_convert_subset(d):
    """
    Quick conversion of windowed data to numpy
    :param d:
    :return:
    """
    l = []
    y = []
    for i in range(len(d.indices)):
        l.append(np.asarray(d.dataset[i]))
        y.append(np.asarray(d.dataset.window_label[i]))
    return np.asarray(l), np.asarray(y)


def jiri_get_descr(data):
    """
    Generates list of tuples. Each Tuple is ('discrete', cardinality). Here, we are trying to figure out the cardinalilty
    :param data:
    :return:
    """
    descr = []
    for f in data.dataset.trans_table.columns:
        descr.append(('discrete', len(data.dataset.vocab.token2id[f])))
    return descr, [s for s in data.dataset.trans_table.columns]


def jiri_assemble_data(train, eval, test):
    """
    Creates a dict with train, dev, test partitions ready to be dumped as hdf5
    :param train:
    :param eval:
    :param test:
    :return:
    """
    D = {}
    descr, colnames = jiri_get_descr(train)
    cols = list(range(1, 11))
    x1, y1 = jiri_convert_subset(train)
    x2, y2 = jiri_convert_subset(eval)
    x3, y3 = jiri_convert_subset(test)
    # need to remove bias - downstream code requires initial indexes to be 0
    minima = np.min(np.min(np.vstack([x1, x2, x3]), axis=0), axis=0)
    x1 = x1 - minima
    x2 = x2 - minima
    x3 = x3 - minima
    nlabels = len(set(y1))
    D['Y_train_descr'] = [('discrete', nlabels)]
    D['X_train'] = x1[..., cols]
    D['Y_train'] = y1
    D['X_dev'] = x2[..., cols]
    D['Y_dev'] = y2
    x, y = jiri_convert_subset(test)
    D['X_test'] = x3[..., cols]
    D['Y_test'] = y3
    D['X_train_descr'] = [descr[i] for i in cols]
    D['descr_column_names'] = [colnames[i] for i in cols]
    return D


def main(args):
    # random seeds
    seed = args.seed
    random.seed(seed)  # python 
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # torch.cuda

    if args.data_type == 'card':
        dataset = TransactionDataset(root=args.data_root,
                                     fname=args.data_fname,
                                     fextension=args.data_extension,
                                     vocab_dir=args.output_dir,
                                     nrows=args.nrows,
                                     user_ids=args.user_ids,
                                     mlm=args.mlm,
                                     cached=args.cached,
                                     stride=args.stride,
                                     flatten=args.flatten,
                                     return_labels=False,
                                     skip_user=args.skip_user)
    elif args.data_type == 'prsa':
        dataset = PRSADataset(stride=args.stride,
                              mlm=args.mlm,
                              return_labels=False,
                              use_station=False,
                              flatten=args.flatten,
                              vocab_dir=args.output_dir)

    else:
        raise Exception(f"data type '{args.data_type}' not defined")

    vocab = dataset.vocab
    custom_special_tokens = vocab.get_special_tokens()

    # split dataset into train, val, test [0.6. 0.2, 0.2]
    totalN = len(dataset)
    trainN = int(0.6 * totalN)

    valtestN = totalN - trainN
    valN = int(valtestN * 0.5)
    testN = valtestN - valN

    assert totalN == trainN + valN + testN

    lengths = [trainN, valN, testN]

    log.info(f"# lengths: train [{trainN}]  valid [{valN}]  test [{testN}]")
    log.info("# lengths: train [{:.2f}]  valid [{:.2f}]  test [{:.2f}]".format(trainN / totalN, valN / totalN,
                                                                               testN / totalN))

    train_dataset, eval_dataset, test_dataset = random_split_dataset(dataset, lengths)

    if args.jiri_just_dump_files:
        D = jiri_assemble_data(train_dataset, eval_dataset, test_dataset)
        fn = "jiri_data.h5"
        with h5.File(fn, 'w') as of:
            print("INFO: Saving jiri stuff in ", fn)
            for key in D.keys():
                if 'descr' in key:
                    saux = json.dumps(D[key])
                    of.create_dataset(key, data=saux)
                # elif 'info' in key:
                #     xaux = np.fromstring(pickle.dumps(newD[key]), dtype='uint8')
                #     of.create_dataset(key, data=xaux)
                else:
                    if D[key] is not None and len(D[key]) > 0:
                        of.create_dataset(key, data=D[key])
        return

    if args.lm_type == "bert":
        tab_net = TabFormerBertLM(custom_special_tokens,
                               vocab=vocab,
                               field_ce=args.field_ce,
                               flatten=args.flatten,
                               ncols=dataset.ncols,
                               field_hidden_size=args.field_hs
                               )
    else:
        tab_net = TabFormerGPT2(custom_special_tokens,
                             vocab=vocab,
                             field_ce=args.field_ce,
                             flatten=args.flatten,
                             )

    log.info(f"model initiated: {tab_net.model.__class__}")

    if args.flatten:
        collactor_cls = "DataCollatorForLanguageModeling"
    else:
        collactor_cls = "TransDataCollatorForLanguageModeling"

    log.info(f"collactor class: {collactor_cls}")
    data_collator = eval(collactor_cls)(
        tokenizer=tab_net.tokenizer, mlm=args.mlm, mlm_probability=args.mlm_prob
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.num_train_epochs,  # total number of training epochs
        logging_dir=args.log_dir,  # directory for storing logs
        save_steps=args.save_steps,
        do_train=args.do_train,
        # do_eval=args.do_eval,
        # evaluation_strategy="epoch",
        prediction_loss_only=True,
        overwrite_output_dir=True,
        # eval_steps=10000
    )

    trainer = Trainer(
        model=tab_net.model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if args.checkpoint:
        model_path = join(args.output_dir, f'checkpoint-{args.checkpoint}')
    else:
        model_path = args.output_dir

    trainer.train(model_path=model_path)


if __name__ == "__main__":

    parser = define_main_parser()
    opts = parser.parse_args()

    opts.log_dir = join(opts.output_dir, "logs")
    makedirs(opts.output_dir, exist_ok=True)
    makedirs(opts.log_dir, exist_ok=True)

    if opts.mlm and opts.lm_type == "gpt2":
        raise Exception("Error: GPT2 doesn't need '--mlm' option. Please re-run with this flag removed.")

    if not opts.mlm and opts.lm_type == "bert":
        raise Exception("Error: Bert needs '--mlm' option. Please re-run with this flag included.")

    main(opts)
