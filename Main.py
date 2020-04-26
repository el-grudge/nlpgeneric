import torch.nn as nn
import pandas as pd
from pathlib import Path
from Dataset import ReviewDataset
from utils import *
from argparse import Namespace
from tqdm import tqdm
import torch.optim as optim

if __name__ == '__main__':

    args = Namespace(
        # Data and Path information
        frequency_cutoff=25,  #=5
        model_state_file='model.pth',
        predictor_csv='tweets_with_splits_full.csv',  # ='tweets_with_splits_lite.csv',
        test_csv='test.csv',
        save_dir='model_storage/',
        vectorizer_file='vectorizer.json',
        # Model hyper parameters
        glove_filepath='/home/minasonbol/PycharmProjects/nlpbasicperceptron/.vector_cache/glove.6B.100d.txt', # GLOVE_MODEL
        use_glove=False, # GLOVE_MODEL
        embedding_size=100, # GLOVE_MODEL
        hidden_dim=300,
        num_channels=256,  # =512,
        stop_early=False,
        # Training hyper parameters
        batch_size=128,  # =10,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=20,  #1,
        seed=1337,
        dropout_p=0.1,
        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True,
        expand_filepaths_to_save_dir=True,
        reload_from_files=False,
        classifier_class ='',
        loss_func_str = '',
        with_weights = False,
    )

    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir,
                                            args.vectorizer_file)

        args.model_state_file = os.path.join(args.save_dir,
                                             args.model_state_file)

        print("Expanded filepaths: ")
        print("\t{}".format(args.vectorizer_file))
        print("\t{}".format(args.model_state_file))

    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    print("Using CUDA: {}".format(args.cuda))
    args.device = torch.device("cuda" if args.cuda else "cpu")

    # handle dirs
    handle_dirs(args.save_dir)

    error_list = []
    errors = {'Classifier': None,
              'loss_func': None,
              'Weights': None,
              'Accuracy': None,
              'Precision': None,
              'Recall': None,
              'F1 Score': None}

    for i in ['Perceptron', 'MLP', 'CNN', 'GloVe']:
        for l in [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()]:
            for w in [False, True]:
                args.classifier_class = i
                args.loss_func_str = remove_punctuation(str(l.type(str)))[0]

                # Set seed for reproducibility
                set_seed_everywhere(args.seed, args.cuda)

                if args.reload_from_files:
                    # training from a checkpoint
                    print("Loading dataset and vectorizer")
                    dataset = ReviewDataset.load_dataset_and_load_vectorizer(args)
                else:
                    print("Loading dataset and creating vectorizer")
                    # create dataset and vectorizer
                    dataset = ReviewDataset.load_dataset_and_make_vectorizer(args)
                    dataset.save_vectorizer(args.vectorizer_file)

                vectorizer = dataset.get_vectorizer()

                # Initialization
                if i == 'GloVe':
                    args.use_glove = True  # GLOVE_MODEL

                # GLOVE_MODEL
                # Use GloVe or randomly initialized embeddings
                if args.use_glove:
                    words = vectorizer.predictor_vocab._token_to_idx.keys()
                    embeddings = make_embedding_matrix(glove_filepath=args.glove_filepath,
                                                       words=words)
                    print("Using pre-trained embeddings")
                else:
                    print("Not using pre-trained embeddings")
                    embeddings = None

                # Classifier
                dimensions = {
                    'input_dim': len(vectorizer.predictor_vocab),
                    'hidden_dim': args.hidden_dim,
                    'output_dim': len(vectorizer.target_vocab),
                    'dropout_p': args.dropout_p,  # GLOVE_MODEL
                    'pretrained_embeddings': embeddings,  # GLOVE_MODEL
                    'padding_idx': 0  # GLOVE_MODEL
                }

                classifier = NLPClassifier(args, dimensions)
                loss_func = l
                if w:
                    if args.loss_func_str == 'BCEWithLogitsLoss':
                        loss_func.pos_weight = pos_weight(dataset.train_df.target.value_counts(), args.batch_size)
                    elif args.loss_func_str == 'CrossEntropyLoss':
                        loss_func.weight = dataset.class_weights
                else:
                    loss_func.weight = None
                optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                                 mode='min', factor=0.5,
                                                                 patience=1)

                train_state = make_train_state(args)

                epoch_bar = tqdm(desc='training routine',
                                 total=args.num_epochs,
                                 position=0)

                dataset.set_split('train')
                train_bar = tqdm(desc='split=train',
                                 total=dataset.get_num_batches(args.batch_size),
                                 position=1,
                                 leave=True)
                dataset.set_split('val')
                val_bar = tqdm(desc='split=val',
                               total=dataset.get_num_batches(args.batch_size),
                               position=1,
                               leave=True)

                # Training loop
                train_state = training_val_loop(args, train_state, dataset, classifier, loss_func, optimizer, scheduler,
                                                train_bar, val_bar, epoch_bar)

                # Application
                classifier.load_state_dict(torch.load(train_state['model_filename']))
                classifier = classifier.to(args.device)

                test_predictor = pd.read_csv(Path().joinpath('data', args.test_csv))

                results = []
                for _, value in test_predictor.iterrows():
                    prediction = predict_target(args,
                                                value['text'],
                                                classifier,
                                                vectorizer,
                                                decision_threshold=0.5)
                    results.append([value['id'], 0 if prediction == 'fake' else 1])

                results = pd.DataFrame(results, columns=['id', 'target'])
                results.to_csv(Path().joinpath('data', '{}_{}_{}_results.csv'.format(i, w, args.loss_func_str)), index=False)

                tn, fp, fn, tp = train_state['test_confusion_matrix']
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1_score = 2 * ((precision*recall)/(precision+recall))

                errors = {'Classifier': i,
                          'loss_func': args.loss_func_str,
                          'Weights': w,
                          'Accuracy': train_state['test_acc'],
                          'Precision': precision,
                          'Recall':  recall,
                          'F1 Score': f1_score
                          }
                error_list.append(errors)

    print(pd.DataFrame(error_list))