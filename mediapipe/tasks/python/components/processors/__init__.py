# Copyright 2022 The MediaPipe Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MediaPipe Tasks Components Processors API."""

import mediapipe.tasks.python.components.processors.classifier_options
import mediapipe.tasks.python.components.processors.embedder_options

ClassifierOptions = classifier_options.ClassifierOptions
EmbedderOptions = embedder_options.EmbedderOptions

# Remove unnecessary modules to avoid duplication in API docs.
del classifier_options
del embedder_options
del mediapipe
