from __future__ import unicode_literals, print_function

import argparse
import json
import random
import re
import warnings
from typing import List

import spacy
import numpy as np
import pandas as pd
from wasabi import Printer
from spacy.gold import GoldParse

import mlflow
import mlflow.pyfunc as mlflow_pyfunc
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec


import mlflow.azureml
from azureml.core import Workspace

warnings.filterwarnings("ignore")
msg = Printer()

# Azure setup

ws = Workspace.from_config()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
experiment_name = 'ner_spacy_mlflow_azure_demo'
mlflow.set_experiment(experiment_name)


class spacyNERModel(mlflow.pyfunc.PythonModel):

    def __init__(self, nlpModel):
        self.nlpModel = nlpModel

    def predict(self, context, model_input):
        # df = pd.read_json(model_input)
        res = model_input['text'].apply(lambda x: str(list(self.nlpModel(x).ents)))
        return res.values


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

    # train_file = './data/train_data.json'
    # model_name = 'spacyNERmodel'
    # iteration = 10

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

        input_schema = Schema([

            ColSpec("string", 'text')
        ])

        output_schema = Schema([
            ColSpec("string", 'predictions')
        ])

        nlp_pyfunc_model = spacyNERModel(nlp)

        mlflow_pyfunc.log_model(python_model=nlp_pyfunc_model, artifact_path='model', conda_env='envs/config'
                                                                                                '/conda_env.yaml',
                                signature=
                                ModelSignature(inputs=input_schema, outputs=output_schema))

        # Model registry
        model_uri = "runs:/{run_id}/{artifact_path}".format(
            run_id=mlflow.active_run().info.run_id,
            artifact_path="model")

        print(model_uri)

        mlflow.register_model(model_uri, 'spacyNERmodel')
