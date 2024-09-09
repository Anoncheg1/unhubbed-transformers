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
- tokenizers (commented) - https://github.com/huggingface/tokenizers
  - Provides “Fast” Rust implementations of today's most used tokenizers.
  - Big amount of Rust Carge open-source dependencies.
- "huggingface-hub"(commented) - communication with HF site.
- safetensors - HF model format for deep learning models, essential dependency.
  - https://github.com/huggingface/safetensors
  - Big amount of Rust Carge open-source (in theory) dependencies.
- "accelerate" (commented) - simplify processes of training at devices and nodes.
  - https://github.com/huggingface/accelerate
- "pyyaml" (commented) - used for the model cards metadata, YAML parser-framework.
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
 README.md                                          |  349 +-------------
 setup.py                                           |    2
 src/transformers/configuration_utils.py            |   28 +
 src/transformers/dependency_versions_check.py      |    8
 src/transformers/dynamic_module_utils.py           |   20 -
 src/transformers/feature_extraction_utils.py       |   16 -
 src/transformers/file_utils.py                     |   40 +-
 src/transformers/generation/configuration_utils.py |   12
 src/transformers/modeling_utils.py                 |   46 +-
 src/transformers/models/__init__.py                |  516 ++++++++++----------
 src/transformers/models/auto/auto_factory.py       |    8
 src/transformers/models/auto/configuration_auto.py |    8
 src/transformers/models/auto/tokenization_auto.py  |   10
 src/transformers/safetensors_conversion.py         |   54 +-
 src/transformers/tokenization_utils_base.py        |   35 +
 src/transformers/utils/__init__.py                 |   50 +-
 src/transformers/utils/hub.py                      |  231 ++++++++-
 src/transformers/utils/logging.py                  |    5
 src/transformers/utils/peft_utils.py               |   39 +-
 19 files changed, 685 insertions(+), 792 deletions(-)
```
