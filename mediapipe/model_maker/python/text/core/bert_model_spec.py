# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Specification for a BERT model."""

import dataclasses
from typing import Dict

from mediapipe.model_maker.python.core import hyperparameters as hp
from mediapipe.model_maker.python.text.core import bert_model_options

_DEFAULT_TFLITE_INPUT_NAME = {
    'ids': 'serving_default_input_word_ids:0',
    'mask': 'serving_default_input_mask:0',
    'segment_ids': 'serving_default_input_type_ids:0'
}


@dataclasses.dataclass
class BertModelSpec:
  """Specification for a BERT model.

  See https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding) for more details.

    Attributes:
      hparams: Hyperparameters used for training.
      model_options: Configurable options for a BERT model.
      do_lower_case: boolean, whether to lower case the input text. Should be
        True / False for uncased / cased models respectively, where the models
        are specified by the `uri`.
      tflite_input_name: Dict, input names for the TFLite model.
      uri: URI for the BERT module.
      name: The name of the object.
  """

  hparams: hp.BaseHParams = hp.BaseHParams(
      epochs=3,
      batch_size=32,
      learning_rate=3e-5,
      distribution_strategy='mirrored')
  model_options: bert_model_options.BertModelOptions = (
      bert_model_options.BertModelOptions())
  do_lower_case: bool = True
  tflite_input_name: Dict[str, str] = dataclasses.field(
      default_factory=lambda: _DEFAULT_TFLITE_INPUT_NAME)
  uri: str = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1'
  name: str = 'Bert'
