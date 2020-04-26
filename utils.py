import os
import re
import string
import numpy as np
import torch
from torch.utils.data import DataLoader
from Perceptron import Perceptron
from MLP import MLPClassifier
from CNN import CNNClassifier
from GloVeClassifier import GloVeClassifier
from sklearn.metrics import confusion_matrix


def make_train_state(args):
    return {'stop_early': args.stop_early,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'train_confusion_matrix': [],
            'val_loss': [],
            'val_acc': [],
            'val_confusion_matrix': [],
            'test_loss': -1,
            'test_acc': -1,
            'test_confusion_matrix': [],
            'test_targets': [],
            'test_predictions': [],
            'model_filename': args.model_state_file}


def update_train_state(args, model, train_state):
    """Handle the training state updates.

        Components:
         - Early Stopping: Prevent overfitting.
         - Model Checkpoint: Model is saved if the model is better

        :param args: main arguments
        :param model: model to train
        :param train_state: a dictionary representing the training state values
        :returns:
            a new train_state
        """

    # Save one model at least
    if train_state['epoch_index'] == 0:
        torch.save(model.state_dict(), train_state['model_filename'])
        train_state['stop_early'] = False

    # Save model if performance improved
    elif train_state['epoch_index'] >= 1:
        loss_tm1, loss_t = train_state['val_loss'][-2:]

        # If loss worsened
        if loss_t >= train_state['early_stopping_best_val']:
            # Update step
            train_state['early_stopping_step'] += 1
        # Loss decreased
        else:
            # Save the best model
            if loss_t < train_state['early_stopping_best_val']:
                torch.save(model.state_dict(), train_state['model_filename'])

            # Reset early stopping step
            train_state['early_stopping_step'] = 0

        # Stop early ?
        train_state['stop_early'] = \
            train_state['early_stopping_step'] >= args.early_stopping_criteria

    return train_state


def format_target(loss_func, target):
    return target.float() if loss_func == 'BCEWithLogitsLoss' else target


def compute_confusion_matrix(loss_func, y_pred, y_target):
    y_target = y_target.cpu()
    if loss_func == 'BCEWithLogitsLoss':
        y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long() #BCEWithLogitsLoss
    else:
        y_pred_indices = y_pred.max(dim=1)[1] #CrossEntropyLoss

    return confusion_matrix(y_target, y_pred_indices).ravel()


def compute_accuracy(loss_func, y_pred, y_target):
    """Predict the target of a predictor"""
    y_target = y_target.cpu()
    if loss_func == 'BCEWithLogitsLoss':
        y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # BCEWithLogitsLoss
    else:
        y_pred_indices = y_pred.max(dim=1)[1]

    n_correct = torch.eq(y_pred_indices, y_target).sum().item()
    return n_correct / len(y_pred_indices) * 100


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def preprocess_text(text):
    """Remove numbers and special charachters from input"""
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def predict_target(args, predictor, classifier, vectorizer, decision_threshold=0.5):
    """Predict the target of a predictor for Perceptron

        Args:
            predictor (str): the text of the predictor
            classifier (Perceptron): the trained model
            vectorizer (ReviewVectorizer): the corresponding vectorizer
            decision_threshold (float): The numerical boundary which separates the target classes
            :param classifier_class: classifier class
    """
    predictor = preprocess_text(predictor)

    if args.classifier_class == 'Perceptron':
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, args.classifier_class))
        result = classifier(args.loss_func_str, vectorized_predictor.view(1, -1))

        if args.loss_func_str == 'BCEWithLogitsLoss':
            probability_value = torch.sigmoid(result).item()
        else:
            probability_value = torch.sigmoid(result.max(dim=1).values).item()  # CrossEntropyLoss

        index = 1
        if probability_value < decision_threshold:
            index = 0

    elif args.classifier_class == 'MLP':
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, args.classifier_class)).view(1, -1)

        if args.loss_func_str == 'BCEWithLogitsLoss':
            result = classifier(args.loss_func_str, vectorized_predictor.view(1, -1))  # BCEWithLogitsLoss
            probability_value = torch.sigmoid(result).item()  # BCEWithLogitsLoss
            index = 1
            if probability_value < decision_threshold:
                index = 0
        else:
            result = classifier(args.loss_func_str, vectorized_predictor, apply_softmax=True)  # CrossEntropyLoss
            probability_value, indices = result.max(dim=1)  # CrossEntropyLoss
            index = indices.item()

    else:
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, args.classifier_class)).unsqueeze(0)

        if args.loss_func_str == 'BCEWithLogitsLoss':
            result = classifier(args.loss_func_str, vectorized_predictor)  # BCEWithLogitsLoss
            probability_value = torch.sigmoid(result).item()  # BCEWithLogitsLoss
            index = 1
            if probability_value < decision_threshold:
                index = 0
        else:
            result = classifier(args.loss_func_str, vectorized_predictor, apply_softmax=True)
            probability_values, indices = result.max(dim=1)
            index = indices.item()

    return vectorizer.target_vocab.lookup_index(index)


def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
        A generator function which wraps the PyTorch DataLoader. It will
          ensure each tensor is on the write device location.
        """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def training_val_loop(args, train_state, dataset, classifier, loss_func, optimizer, scheduler, train_bar, val_bar,
                      epoch_bar):
    """Performs the training-validation loop

    Args:
        args: main arguments
        train_state: a dictionary representing the training state values
        dataset (Dataset): the dataset
        classifier (Classifer): an instance of the classifier
        loss_func: loss function
        optimizer: optimizer function
        scheduler:
        train_bar: tqdm bar to track progress
        val_bar: tqdm bar to track progress
        epoch_bar: tqdm bar to track progress

    Returns:
        train_state: a dictionary with the updated training state values
    """
    try:
        for epoch_index in range(args.num_epochs):
            train_state['epoch_index'] = epoch_index

            # Iterate over training dataset

            # setup: batch generator, set loss and acc to 0, set train mode on
            dataset.set_split('train')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.0
            running_acc = 0.0
            classifier.train()

            for batch_index, batch_dict in enumerate(batch_generator):
                # the training routine is these 5 steps:

                # --------------------------------------
                # step 1. zero the gradients
                optimizer.zero_grad()

                # step 2. compute the output
                y_pred = classifier(args.loss_func_str, x_in=batch_dict['x_data'])

                # step 3. compute the loss
                target = format_target(args.loss_func_str, batch_dict['y_target'])
                loss = loss_func(y_pred, target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(args.loss_func_str, y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                # update bar
                train_bar.set_postfix(loss=running_loss,
                                      acc=running_acc,
                                      epoch=epoch_index)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            train_state['train_acc'].append(running_acc)

            # Iterate over val dataset
            # setup: batch generator, set loss and acc to 0; set eval mode on
            dataset.set_split('val')
            batch_generator = generate_batches(dataset,
                                               batch_size=args.batch_size,
                                               device=args.device)
            running_loss = 0.
            running_acc = 0.
            classifier.eval()

            for batch_index, batch_dict in enumerate(batch_generator):
                # compute the output
                y_pred = classifier(args.loss_func_str, x_in=batch_dict['x_data'])

                # compute the loss
                target = format_target(args.loss_func_str, batch_dict['y_target'])
                loss = loss_func(y_pred, target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(args.loss_func_str, y_pred, batch_dict['y_target'])
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                val_bar.set_postfix(loss=running_loss,
                                    acc=running_acc,
                                    epoch=epoch_index)
                val_bar.update()

            train_state['val_loss'].append(running_loss)
            train_state['val_acc'].append(running_acc)

            train_state = update_train_state(args=args, model=classifier,
                                             train_state=train_state)

            scheduler.step(train_state['val_loss'][-1])

            if train_state['stop_early']:
                break

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()
    except KeyboardInterrupt:
        print("Exiting loop")

    # compute the loss & accuracy on the test set using the best available model
    classifier.load_state_dict(torch.load(train_state['model_filename']))
    classifier = classifier.to(args.device)

    dataset.set_split('test')
    batch_generator = generate_batches(dataset,
                                       batch_size=args.batch_size,
                                       device=args.device)
    running_loss = 0.0
    running_acc = 0.0
    running_tn = 0.0
    running_fp = 0.0
    running_fn = 0.0
    running_tp = 0.0
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(args.loss_func_str, x_in=batch_dict['x_data'])

        # compute the loss
        target = format_target(args.loss_func_str, batch_dict['y_target'])
        loss = loss_func(y_pred, target)
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(args.loss_func_str, y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)
        # compute the confusion matrix
        tn, fp, fn, tp = compute_confusion_matrix(args.loss_func_str, y_pred, batch_dict['y_target'])
        running_tn += (tn - running_tn) / (batch_index + 1)
        running_fp += (fp - running_fp) / (batch_index + 1)
        running_fn += (fn - running_fn) / (batch_index + 1)
        running_tp += (tp - running_tp) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc
    train_state['test_confusion_matrix'] = [running_tn, running_fp, running_fn, running_tp]
    train_state['test_targets'] = batch_dict['y_target']
    train_state['test_predictions'] = y_pred

    return train_state


def NLPClassifier(args, dimensions):
    """Builds a classifier

    Args:
        args: main arguments
        classifier_class: classifier class to be defined
        dimensions: neural network dimensions
        loss_func: loss function to be used

    Returns:
        classifier: built classfier
        loss_func: loss function
        optimizer: optimizer
        scheduler
    """
    if args.classifier_class == 'Perceptron':
        classifier = Perceptron(num_features=dimensions['input_dim'], loss_func=args.loss_func_str)

    elif args.classifier_class == 'MLP':
        classifier = MLPClassifier(input_dim=dimensions['input_dim'],
                                   hidden_dim=dimensions['hidden_dim'],
                                   output_dim=dimensions['output_dim'],
                                   loss_func=args.loss_func_str)

    elif args.classifier_class == 'CNN':
        classifier = CNNClassifier(initial_num_channels=dimensions['input_dim'],
                      num_classes=dimensions['output_dim'],
                      num_channels=args.num_channels,
                      loss_func=args.loss_func_str)

    # GLOVE_MODEL
    elif args.classifier_class == 'GloVe':
        classifier = GloVeClassifier(embedding_size=args.embedding_size,
                                    num_embeddings=dimensions['input_dim'],
                                    num_channels=args.num_channels,
                                    hidden_dim=args.hidden_dim,
                                    num_classes=dimensions['output_dim'],
                                    dropout_p=args.dropout_p,
                                    pretrained_embeddings=dimensions['pretrained_embeddings'],
                                    padding_idx=0,
                                    loss_func=args.loss_func_str)

    classifier = classifier.to(args.device)

    return classifier


def remove_punctuation(s: str):
    return [x for x in ''.join(char for char in s if char not in string.punctuation).split()]

# GLOVE_MODEL
def load_glove_from_file(glove_filepath):
    """
    Load the GloVe embeddings

    Args:
        glove_filepath (str): path to the glove embeddings file
    Returns:
        word_to_index (dict), embeddings (numpy.ndarary)
    """

    word_to_index = {}
    embeddings = []
    with open(glove_filepath, "r") as fp:
        for index, line in enumerate(fp):
            line = line.split(" ")  # each line: word num1 num2 ...
            word_to_index[line[0]] = index  # word = line[0]
            embedding_i = np.array([float(val) for val in line[1:]])
            embeddings.append(embedding_i)
    return word_to_index, np.stack(embeddings)


def make_embedding_matrix(glove_filepath, words):
    """
    Create embedding matrix for a specific set of words.

    Args:
        glove_filepath (str): file path to the glove embeddigns
        words (list): list of words in the dataset
    """
    word_to_idx, glove_embeddings = load_glove_from_file(glove_filepath)
    embedding_size = glove_embeddings.shape[1]

    final_embeddings = np.zeros((len(words), embedding_size))

    for i, word in enumerate(words):
        if word in word_to_idx:
            final_embeddings[i, :] = glove_embeddings[word_to_idx[word]]
        else:
            embedding_i = torch.ones(1, embedding_size)
            torch.nn.init.xavier_uniform_(embedding_i)
            final_embeddings[i, :] = embedding_i

    return final_embeddings


def pos_weight(class_counts, batch_size):
    pos_class = class_counts['real']
    neg_class = class_counts['fake']
    return torch.as_tensor([neg_class/pos_class]*batch_size, dtype=float)