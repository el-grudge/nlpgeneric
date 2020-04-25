import torch.nn as nn
import pandas as pd
from pathlib import Path
from Dataset import ReviewDataset
from utils import *
from argparse import Namespace
from tqdm import tqdm

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
        # Training hyper parameters
        batch_size=128,  # =10,
        early_stopping_criteria=5,
        learning_rate=0.001,
        num_epochs=10,  #1,
        seed=1337,
        dropout_p=0.1,
        # Runtime options
        catch_keyboard_interrupt=True,
        cuda=True,
        expand_filepaths_to_save_dir=True,
        reload_from_files=False,
        classifier_class='' #'CNN', #'MLP', #'Perceptron', #'GloVe'
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

    errors = {}
    #for i in ['Perceptron', 'MLP', 'CNN', 'GloVe']:
    for i in ['CNN', 'MLP', 'Perceptron']:
        args.classifier_class = i

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
        # check why BCE doesn't work with CNN and MLP
        #loss_func = nn.BCEWithLogitsLoss() if args.classifier_class == 'Perceptron' else nn.CrossEntropyLoss()  # dataset.class_weights
        loss_func = nn.CrossEntropyLoss()
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

        # print("Training loss: {:.3f}".format(train_state['train_loss'][0]))
        # print("Training Accuracy: {:.2f}".format(train_state['train_acc'][0]))
        # print("Training Confusion Matrix: ", train_state['train_confusion_matrix'])

        # print("Validation loss: {:.3f}".format(train_state['val_loss'][0]))
        # print("Validation Accuracy: {:.2f}".format(train_state['val_acc'][0]))
        # print("Validation Confusion Matrix: ", train_state['val_confusion_matrix'])

        # print("{} Test loss: {}".format(args.classifier_class, train_state['test_loss']))
        # print("Test Accuracy: {:.2f}".format(train_state['test_acc']))
        # print("Test Confusion Matrix: ", train_state['test_confusion_matrix'])
        errors[i] = train_state['test_acc']

        # Application
        classifier.load_state_dict(torch.load(train_state['model_filename']))
        classifier = classifier.to(args.device)

        test_predictor = pd.read_csv(Path().joinpath('data', args.test_csv))

        results = []
        for _, value in test_predictor.iterrows():
            prediction = predict_target(args.classifier_class, value['text'], classifier, vectorizer,
                                        decision_threshold=0.5)
            results.append([value['id'], 0 if prediction == 'fake' else 1])

        results = pd.DataFrame(results, columns=['id', 'target'])
        results.to_csv(Path().joinpath('data', '{}_results.csv'.format(i)), index=False)

    print(errors)

    '''
    # Inference
    test_predictor = "fires are running wild"

    classifier = classifier.cpu()
    prediction = predict_target(test_predictor, classifier, vectorizer, decision_threshold=0.5)
    print("{} -> {}".format(test_predictor, prediction))

    # Interpretability
    classifier.fc1.weight.shape
    # Sort weights
    fc1_weights = classifier.fc1.weight.detach()[0]
    _, indices = torch.sort(fc1_weights, dim=0, descending=True)
    indices = indices.numpy().tolist()

    # Top 20 words
    print("Influential words in Positive Reviews:")
    print("--------------------------------------")
    for i in range(20):
        print(vectorizer.predictor_vocab.lookup_index(indices[i]))

    print("====\n\n\n")

    # Top 20 negative words
    print("Influential words in Negative Reviews:")
    print("--------------------------------------")
    indices.reverse()
    for i in range(20):
        print(vectorizer.predictor_vocab.lookup_index(indices[i]))
    '''