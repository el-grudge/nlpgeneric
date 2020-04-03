import os
import re
import string
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from Perceptron import Perceptron
from MLP import MLPClassifier
from CNN import CNNClassifier


def make_train_state(args):
    return {'stop_early': False,
            'early_stopping_step': 0,
            'early_stopping_best_val': 1e8,
            'learning_rate': args.learning_rate,
            'epoch_index': 0,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': -1,
            'test_acc': -1,
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


def format_target(classifier_class, target):
    return target.float() if classifier_class == 'Perceptron' else target


def compute_accuracy(classifier, y_pred, y_target):
    """Predict the target of a predictor"""
    y_target = y_target.cpu()
    if classifier == 'Perceptron':
        y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()
    elif classifier == 'MLP':
        _, y_pred_indices = y_pred.max(dim=1)
    elif classifier == 'CNN':
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
    text = text.lower()
    text = re.sub(r"([.,!?])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z.,!?]+", r" ", text)
    return text


def predict_target(classifier_class, predictor, classifier, vectorizer, decision_threshold=0.5):
    """Predict the target of a predictor for Perceptron

        Args:
            predictor (str): the text of the predictor
            classifier (Perceptron): the trained model
            vectorizer (ReviewVectorizer): the corresponding vectorizer
            decision_threshold (float): The numerical boundary which separates the target classes
            :param classifier_class: classifier class
        """
    predictor = preprocess_text(predictor)

    if classifier_class == 'Perceptron':
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, classifier_class))
        result = classifier(vectorized_predictor.view(1, -1))

        probability_value = torch.sigmoid(result).item()
        index = 1
        if probability_value < decision_threshold:
            index = 0
    elif classifier_class == 'MLP':
        vectorized_predictor = torch.tensor(vectorizer.vectorize(predictor, classifier_class)).view(1, -1)
        result = classifier(vectorized_predictor, apply_softmax=True)

        probability_value, indices = result.max(dim=1)
        index = indices.item()
    elif classifier_class == 'CNN':
        vectorized_predictor = vectorizer.vectorize(predictor, classifier_class)
        vectorized_predictor = torch.tensor(vectorized_predictor).unsqueeze(0)
        result = classifier(vectorized_predictor, apply_softmax=True)

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
                y_pred = classifier(x_in=batch_dict['x_data'].float())

                # step 3. compute the loss
                target = format_target(args.classifier_class, batch_dict['y_target'])
                loss = loss_func(y_pred, target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # step 4. use loss to produce gradients
                loss.backward()

                # step 5. use optimizer to take gradient step
                optimizer.step()
                # -----------------------------------------
                # compute the accuracy
                acc_t = compute_accuracy(args.classifier_class, y_pred, batch_dict['y_target'])
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
                y_pred = classifier(x_in=batch_dict['x_data'].float())

                # compute the loss
                target = format_target(args.classifier_class, batch_dict['y_target'])
                loss = loss_func(y_pred, target)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / (batch_index + 1)

                # compute the accuracy
                acc_t = compute_accuracy(args.classifier_class, y_pred, batch_dict['y_target'])
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

            train_bar.n = 0
            val_bar.n = 0
            epoch_bar.update()

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
    classifier.eval()

    for batch_index, batch_dict in enumerate(batch_generator):
        # compute the output
        y_pred = classifier(x_in=batch_dict['x_data'].float())

        # compute the loss
        target = format_target(args.classifier_class, batch_dict['y_target'])
        loss = loss_func(y_pred, target)
        loss_t = loss.item()
        running_loss += (loss_t - running_loss) / (batch_index + 1)

        # compute the accuracy
        acc_t = compute_accuracy(args.classifier_class, y_pred, batch_dict['y_target'])
        running_acc += (acc_t - running_acc) / (batch_index + 1)

    train_state['test_loss'] = running_loss
    train_state['test_acc'] = running_acc

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
        classifier = Perceptron(num_features=dimensions['input_dim'])
    elif args.classifier_class == 'MLP':
        classifier = MLPClassifier(input_dim=dimensions['input_dim'],
                                   hidden_dim=dimensions['hidden_dim'],
                                   output_dim=dimensions['output_dim'])
    elif args.classifier_class == 'CNN':
        classifier = CNNClassifier(initial_num_channels=dimensions['input_dim'],
                      num_classes=dimensions['output_dim'],
                      num_channels=args.num_channels)


    classifier = classifier.to(args.device)
    loss_func = args.loss_func
    optimizer = optim.Adam(classifier.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                     mode='min', factor=0.5,
                                                     patience=1)

    return classifier, loss_func, optimizer, scheduler


def remove_punctuation(s: str):
    return [x for x in ''.join(char for char in s if char not in string.punctuation).split()]