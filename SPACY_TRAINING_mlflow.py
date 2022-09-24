from __future__ import unicode_literals, print_function

import argparse
import json
import random
import re
import warnings
from typing import List

import spacy
import numpy as np
from wasabi import Printer
from spacy.gold import GoldParse

import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema


warnings.filterwarnings("ignore")
msg = Printer()


def trim_entity_spans(data: List) -> List:
    invalid_span_tokens = re.compile(r"\s")
    cleaned_data = []
    for jsondata in data:
        entities = jsondata["entities"]
        text = jsondata["content"].lower()
        valid_entities = []
        try:
            aa = 0
            for ent in entities:
                lbltxt = ent['labelText'].lower()
                lblInd = text.find(lbltxt, aa)
                valid_start = lblInd
                valid_end = lblInd + len(lbltxt)
                aa = valid_end
                while valid_start < len(text) and invalid_span_tokens.match(
                        text[valid_start]
                ):
                    valid_start += 1
                while valid_end > 1 and invalid_span_tokens.match(text[valid_end - 1]):
                    valid_end -= 1

                if ent['labelText'].lower() in text[valid_start:valid_end]:
                    valid_entities.append([valid_start, valid_end, ent['labelName'].lower()])
            cleaned_data.append([text.lower(), {"entities": valid_entities}])
        except:
            msg.good("Warning: Entity not found ".format(text))

    return cleaned_data


def get_trained_spacy_model(data, iterations, model=None):
    TRAIN_DATA = data
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for item in TRAIN_DATA:
        for ent in item['entities']:
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for item in TRAIN_DATA:
                text = nlp.make_doc(item['content'])  # <--- add this
                gold = GoldParse(text, entities=item['entities'])
                nlp.update(
                    [text],  # batch of texts
                    [gold],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument('--json_train_file', type=str, help='train file in json format')
    parser.add_argument('--output_model_name', type=str, help='output model name')
    parser.add_argument('--iteration', type=int, help='no of iteration to train over')

    args = parser.parse_args()

    train_file = args.json_train_file
    model_name = args.output_model_name
    iteration = args.iteration

    with mlflow.start_run():

        mlflow.log_param('epochs', iteration)

        with open(train_file, 'r') as f:
            TrainData = json.load(f)

        TRAIN_DATA = TrainData
        nlp = spacy.blank('en')  # create blank Language class
        # create the built-in pipeline components and add them to the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if 'ner' not in nlp.pipe_names:
            ner = nlp.create_pipe('ner')
            nlp.add_pipe(ner, last=True)

        # add labels
        for item in TRAIN_DATA:
            for ent in item['entities']:
                ner.add_label(ent[2])

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
        with nlp.disable_pipes(*other_pipes):  # only train NER
            optimizer = nlp.begin_training()
            for itn in range(iteration):
                print("Statring iteration " + str(itn))
                random.shuffle(TRAIN_DATA)
                losses = {}
                for item in TRAIN_DATA:
                    text = nlp.make_doc(item['content'])  # <--- add this
                    gold = GoldParse(text, entities=item['entities'])
                    nlp.update(
                        [text],  # batch of texts
                        [gold],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)

                print(losses)

            mlflow.log_metric('loss', losses['ner'])

        nlp.to_disk('spacyNERmodel')

        mlflow.spacy.log_model(nlp, artifact_path='spacyNERmodel', conda_env='envs/config/conda_env.yaml')

        # spacy_nlp = get_trained_spacy_model(TrainData, iteration)
        # spacy_nlp.to_disk('/model' + model_name)
