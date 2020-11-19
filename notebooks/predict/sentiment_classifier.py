import glob
import os
import pandas as pd
import random
import spacy
from spacy.util import minibatch, compounding

model_destination = "../../resources/spacy"
imdb_dir = "/Users/ericcarlson/Desktop/Datasets/Document Classification/ACL_IMDB"

def load_data(data_dir: str,
              split: float = 0.8,
              max_count: int = 128) -> tuple:
    # spaCy uses  a list of (text, label_dict) tuples to train
    spacy_data = []
    # labels are in sub-directories
    for label in ["pos", "neg"]:
        count = 0
        sub_dir = os.path.join(data_dir, label)

        for fname in glob.iglob(os.path.join(sub_dir, "*.txt")):
            if count >= max_count:
                break

            with open(fname) as textfile:
                text = textfile.read()
                # do some basic pre-processing
                text = text.replace("<br /> ", "\n\n")
                # "cats" refers to categories in spaCy
                label_dict = {
                    "cats": {
                        "P": "pos" == label,
                        "N": "neg" == label
                    }
                }
                spacy_data.append((text, label_dict))
                count += 1

    # shuffle and divide spaCy data
    random.shuffle(spacy_data)
    split_at = int(len(spacy_data) * split)
    # return train and test sets
    return spacy_data[:split_at], spacy_data[split_at:]


def train_model(train_data: list,
                test_data: list,
                epochs: int = 4
                ) -> None:
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        # TODO: test ensemble, simple_cnn, bow
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("P")
    textcat.add_label("N")

    excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]

    with nlp.disable_pipes(excluded_pipes):
        optimizer = nlp.begin_training()
        batch_sizes = compounding(4.0, 32.0, 1.001)

        print("\t".join(["L", "P", "R", "F"]))
        for i in range(epochs):
            print(f"epoch {i}")
            loss = {}
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)

            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text,
                           labels,
                           drop=0.2,
                           sgd=optimizer,
                           losses=loss
                           )
            # Evaluate the model in its current state.
            with textcat.model.use_params(optimizer.averages):
                scores = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(f"{loss['textcat']: .3f}\t"
                      f"{scores['P']: .3f}\t"
                      f"{scores['R']: .3f}\t"
                      f"{scores['F']: .3f}")

        # Save the model.
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(model_destination)


def evaluate_model(tokenizer,
                   textcat,
                   test_data: list
                   ) -> dict:
    texts, labels = zip(*test_data)

    texts = (tokenizer(text) for text in texts)
    tp, fp, tn, fn = 0, 1e-8, 0, 1e-8

    for i, text in enumerate(textcat.pipe(texts)):
        p_label = labels[i]["cats"]["P"]

        for predicted_label, score in text.cats.items():
            # just iterate over the one label, reduce duplication
            if predicted_label == "N":
                continue

            if score >= 0.5 and p_label:
                tp += 1
            elif score >= 0.5 and not p_label:
                fp += 1
            elif score < 0.5 and p_label:
                fn += 1
            elif score < 0.5 and not p_label:
                tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * precision * recall / (precision + recall)

    return {'P': precision, 'R': recall, 'F': f_score}


if __name__ == '__main__':
    max_count_ = 1024
    epochs_ = 16

    train_set, test_set = load_data(data_dir=os.path.join(imdb_dir, "train"), split=0.8, max_count=max_count_)
    train_model(train_set, test_set, epochs=epochs_)

    nlp_ = spacy.load(model_destination)
    tokenizer_ = nlp_.tokenizer
    textcat_ = nlp_.get_pipe("textcat")

    text_ = next(textcat_.pipe([tokenizer_("The movie was bad and I would not recommend it.")]))
    predicted_label_, score_ = text_.cats.items()
    print(predicted_label_, score_)
