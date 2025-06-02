# Dependencies
from file [dependency_versions_check.py](src/transformers/dependency_versions_check.py "dependency_versions_check.py")
- "python",
- regex - for OpenAI GPT
  - https://pypi.org/project/regex/ https://github.com/mrabarnett/mrab-regex/
- tqdm - to print progress bar.
- filelock https://github.com/tox-dev/filelock/ https://pypi.org/project/filelock/
- requests - HTTP requests
- packaging - parse versions
- filelock - filesystem locks, e.g., to prevent parallel downloads
- numpy
- tokenizers - https://github.com/huggingface/tokenizers
  - Provides “Fast” Rust implementations of today's most used tokenizers.
  - Big amount of Rust Carge open-source dependencies.
- "huggingface-hub"(commented) - communication with HF site.
- safetensors - HF model format for deep learning models, essential dependency.
  - https://github.com/huggingface/safetensors
  - Big amount of Rust Carge open-source (in theory) dependencies.
- "accelerate" - simplify processes of training at devices and nodes.
  - https://github.com/huggingface/accelerate
- "pyyaml" - used for the model cards metadata, YAML parser-framework.
# Dependencies installation
```sh
pip install regex tqdm filelock requests packaging filelock numpy
pip install safetensors
```
# Installation
```sh
python setup.py install --user
```
Note: In setup.py dependencies installation from pypi.org was disabled.

Or just copy `src/transformers` to `~/.local/lib/python3.12/site-packages/transformers`.
# Changes
[CHANGES.md](CHANGES.md "CHANGES.md")

<patch.patch>

```text
 setup.py                                           |   4 +-
 src/transformers/__init__.py                       |   4 +-
 src/transformers/audio_utils.py                    |   2 +-
 src/transformers/commands/env.py                   |   2 +-
 src/transformers/configuration_utils.py            |  18 +-
 src/transformers/dynamic_module_utils.py           |   6 +-
 src/transformers/feature_extraction_utils.py       |  18 +-
 src/transformers/file_utils.py                     |  26 +-
 src/transformers/generation/configuration_utils.py |   8 +-
 src/transformers/generation/utils.py               |   2 +-
 src/transformers/image_processing_base.py          |  20 +-
 src/transformers/image_utils.py                    |   2 +-
 src/transformers/integrations/deepspeed.py         |   6 +-
 src/transformers/keras_callbacks.py                |   2 +-
 src/transformers/modelcard.py                      |   6 +-
 src/transformers/modeling_flax_utils.py            |  20 +-
 src/transformers/modeling_tf_utils.py              |   8 +-
 src/transformers/modeling_utils.py                 |  48 +--
 .../models/deprecated/realm/retrieval_realm.py     |   2 +-
 .../checkpoint_reshaping_and_interoperability.py   |   2 +-
 .../models/oneformer/image_processing_oneformer.py |   4 +-
 .../pix2struct/image_processing_pix2struct.py      |   2 +-
 src/transformers/pipelines/__init__.py             |   2 +-
 src/transformers/pipelines/audio_classification.py |   2 +-
 .../pipelines/automatic_speech_recognition.py      |   2 +-
 src/transformers/pipelines/base.py                 |  14 +-
 src/transformers/pipelines/video_classification.py |   2 +-
 .../pipelines/zero_shot_audio_classification.py    |   2 +-
 src/transformers/processing_utils.py               |  62 ++--
 src/transformers/safetensors_conversion.py         |  43 +--
 src/transformers/testing_utils.py                  |   6 +-
 src/transformers/tokenization_utils_base.py        |  48 +--
 src/transformers/trainer.py                        |  15 +-
 src/transformers/training_args.py                  |   2 +-
 src/transformers/utils/__init__.py                 |  44 +--
 src/transformers/utils/attention_visualizer.py     |   2 +-
 src/transformers/utils/chat_template_utils.py      |  12 +-
 src/transformers/utils/constants.py                |  14 +
 src/transformers/utils/hub.py                      | 371 +++++++++++----------
 src/transformers/utils/logging.py                  |   8 +-
 src/transformers/video_processing_utils.py         |  14 +-
 src/transformers/video_utils.py                    |   2 +-
 42 files changed, 450 insertions(+), 429 deletions(-)
```
# Internet request
In convert and text files enconter links that requested by “requests” and passed to “Pillow” library mainly.

Can be found by command:
```grep -RinE "\"https?:" | grep .py | grep -Eo "(http|https)://[a-zA-Z0-9./?=_%:-]*" | sort -u```

Potentially may carry backdoors payloads.
