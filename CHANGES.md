

Changed in form of errors fixing.
Also there is patch file.

## ModuleNotFoundError: No module named 'huggingface_hub'
File "/home/t/.local/lib/python3.12/site-packages/transformers/utils/__init__.py", line 18, 19
- comment:
: from huggingface_hub import get_full_repo_name  # for backward compatibility
: from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY as DISABLE_TELEMETRY  # for backward compatibility

File "/home/t/.local/lib/python3.12/site-packages/transformers/utils/__init__.py", line 68
- comment: from .hub import ...

File "/home/t/.local/lib/python3.12/site-packages/transformers/utils/logging.py", line 35
- comment: import huggingface_hub.utils as hf_hub_utils

File "/home/t/.local/lib/python3.12/site-packages/transformers/utils/logging.py", line 53
- replace: _tqdm_active = not hf_hub_utils.are_progress_bars_disabled()
- with: _tqdm_active = True

File "/home/t/.local/lib/python3.12/site-packages/transformers/utils/peft_utils.py", line 20, in <module>
- comment : from .hub import cached_file
- replace elif with else:      assert os.path.isdir(model_id)
- comment : else.. file:/home/u/sources/transformers/src/transformers/utils/peft_utils.py::87

File "/home/t/.local/lib/python3.12/site-packages/transformers/file_utils.py", line 20
- comment:
  - from huggingface_hub import get_full_repo_name
  - from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY as DISABLE_TELEMETRY

File "/home/t/.local/lib/python3.12/site-packages/transformers/file_utils.py", line 21

File "/home/t/.local/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 37, in <module>
    from .dynamic_module_utils import custom_object_save
- comment: from .dynamic_module_utils import custom_object_save
- comment two lines: if self._auto_class is not None: file:/home/u/sources/transformers/src/transformers/tokenization_utils_base.py::2582

-----------------------
File "/home/t/.local/lib/python3.12/site-packages/transformers/modeling_utils.py", line 37, in <module>
    from huggingface_hub import split_torch_state_dict_into_shards
- comment 37 from huggingface_hub import split_torch_state_dict_into_shards file:/home/u/sources/transformers/src/transformers/modeling_utils.py::37
- comment 45 from .dynamic_module_utils import custom_object_save

-----------
safetensors_conversion.py", line 6
- comment: from huggingface_hub import Discussion, HfApi, get_repo_discussions

NameError: name 'HfApi' is not defined
- comment function: previous_pr, get_conversion_pr_reference

## No package metadata was found for The 'huggingface-hub>=0.23.2,<1.0' distribution was not found and is required by this application.
utils/import_utils.py", line 1659, in _get_module

dependency_versions_check.py

comment:
- # "tokenizers",
- # "huggingface-hub",
- # "accelerate",
- # "pyyaml",
## models/__init__.py ImportError: cannot import name 'CLOUDFRONT_DISTRIB_PREFIX' from 'transformers.utils
disable all except bert

## ImportError: cannot import name 'PushToHubMixin' from 'transformers.utils'
 File "/home/t/.local/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 38
- comment:
  - 41: PushToHubMixin
  - 48: download_url
- remove PushToHubMixin from class PreTrainedTokenizerBase(SpecialTokensMixin, PushToHubMixin):
  - class PreTrainedTokenizerBase(SpecialTokensMixin): # PushToHubMixin
------------------------------
File "/home/t/.local/lib/python3.12/site-packages/transformers/generation/configuration_utils.py", line 26, in <module>
    from ..utils import (
- comment lines:PushToHubMixin,
    download_url,
---------------------
modeling_utils.py", line 61
- comment file:/home/u/sources/transformers/src/transformers/modeling_utils.py::76
  - PushToHubMixin, download_url, has_file
- exclude file:/home/u/sources/transformers/src/transformers/modeling_utils.py::1298
  - class PreTrainedModel(nn.Module, ModuleUtilsMixin, GenerationMixin, PeftAdapterMixin): # PushToHubMixin
- comment function “push_to_hub” file:/home/u/sources/transformers/src/transformers/modeling_utils.py::2826
- comment lines PreTrainedModel.push_to_hub = copy_ ... file:/home/u/sources/transformers/src/transformers/modeling_utils.py::4720
## AttributeError: type object 'PreTrainedTokenizerBase' has no attribute 'push_to_hub'
File "/home/t/.local/lib/python3.12/site-packages/transformers/tokenization_utils_base.py", line 4230
- comment all bottom lines
## AddedToken
TypeError: unhashable type: 'AddedToken'

file:/home/u/proj-pers-ai/transformers/tokenization_utils_base.py::85

add:
#+begin_src python :results output :exports both :session s1
def __setstate__(self, state):
    self.__dict__.update(state)

def __hash__(self):
    return hash((self.content, self.single_word, self.lstrip, self.rstrip, self.special, self.normalized))

def __eq__(self, other):
    if not isinstance(other, AddedToken):
        return False
    return (self.content, self.single_word, self.lstrip, self.rstrip, self.special, self.normalized) == (
        other.content, other.single_word, other.lstrip, other.rstrip, other.special, other.normalized
    )
#+end_src
## cannot import name 'HF_MODULES_CACHE' from 'transformers.utils
 File "/home/t/.local/lib/python3.12/site-packages/transformers/models/auto/tokenization_auto.py", line 25
- comment 25, 26:
  - from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
  - from ...modeling_gguf_pytorch_utils import load_gguf_checkpoint

----------------------
/home/u/sources/transformers/src/transformers/models/auto/configuration_auto.py
- comment 25: from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code

## No module named 'tokenizers'
*Failed to import transformers.integrations.ggml because of the following error (look up to see its traceback):*
File "/home/t/.local/lib/python3.12/site-packages/transformers/configuration_utils.py", line 29
- comment 28 : from .modeling_gguf_pytorch_utils import load_gguf_checkpoint
- comment 29 : from .dynamic_module_utils import custom_object_save
- comment 3 lines: if gguf_file: file:/home/u/sources/transformers/src/transformers/configuration_utils.py::652
- comment 400 two lines: if self._auto_class is not None:

## cannot import name 'PushToHubMixin' from 'transformers.utils
 File "/home/t/.local/lib/python3.12/site-packages/transformers/configuration_utils.py", line 30
- comment:
  - 32: PushToHubMixin,
  - 37: download_url,

remove: class PretrainedConfig(): # PushToHubMixin file:/home/u/sources/transformers/src/transformers/configuration_utils.py::50

-----------------------
feature_extraction_utils.py", line 29
- comment
  - in “from .utils import (“:
    - PushToHubMixin
    - download_url
  - in FeatureExtractionMixin exclude PushToHubMixin
  - comment last lines in file: FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)

## name 'PushToHubMixin' is not defined
File "/home/t/.local/lib/python3.12/site-packages/transformers/generation/configuration_utils.py", line 71, in <module>

replace: class GenerationConfig(): # PushToHubMixinfile:/home/u/sources/transformers/src/transformers/generation/configuration_utils.py::71
## type object 'PretrainedConfig' has no attribute 'push_to_hub'
file:/home/u/sources/transformers/src/transformers/configuration_utils.py::1110
- comment all bottom lines
## NameError: name 'cached_file' is not defined
cached_file located in utils/hub.py and imported in units/__init__.py

from huggingface_hub constants used:
#+begin_src python :results none :exports code :eval no
# replacement for huggingface_hub.constants
import re

class huggingface_hub:
    class constants:
        HF_HOME = os.path.expanduser(
            os.getenv(
                "HF_HOME",
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
            )
        )
        HF_HUB_OFFLINE=True
        default_cache_path = os.path.join(HF_HOME, "hub")
        HF_HUB_CACHE = default_cache_path
        HF_HUB_DISABLE_TELEMETRY=False
        REGEX_COMMIT_HASH=re.compile(r"^[0-9a-f]{40}$")
constants = huggingface_hub.constants

class GatedRepoError(Exception):
    pass
#+end_src

comment # @_deprecate_method(version="4.39.0", message="This method is outdated and does not support the new cache system.")



##* NameError: name 'try_to_load_from_cache' is not defined
locaded in huggingface_hub/file_download.py:1531:def try_to_load_from_cache(

we add folloing code to transoformers/utils/hub.py
#+begin_src python :results none :exports code :eval no
# Return value when trying to load a file from cache but the file does not exist in the distant repo.
from typing import Any

_CACHED_NO_EXIST = object()
_CACHED_NO_EXIST_T = Any

REPO_TYPE_DATASET = "dataset"
REPO_TYPE_SPACE = "space"
REPO_TYPE_MODEL = "model"
REPO_TYPES = [None, REPO_TYPE_MODEL, REPO_TYPE_DATASET, REPO_TYPE_SPACE]

from typing import Any

def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Union[str, _CACHED_NO_EXIST_T, None]:
    """
    Explores the cache to return the latest cached file for a given revision if\
 found.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo on huggingface.co.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to `"main"` if it's not provided and no `commit_hash` is
            provided either.
        repo_type (`str`, *optional*):
            The type of the repository. Will default to `"model"`.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.

    Example:

    ```python
    from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

    filepath = try_to_load_from_cache()
    if isinstance(filepath, str):
        # file exists and is cached
        ...
    elif filepath is _CACHED_NO_EXIST:
        # non-existence of file is cached
        ...
    else:
        # file is not cached
        ...
    ```
    """
    if revision is None:
        revision = "main"
    if repo_type is None:
        repo_type = "model"
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Invalid repo type: {repo_type}. Accepted repo types are: {str(REPO_TYPES)}")
    if cache_dir is None:
        cache_dir = HF_HUB_CACHE

    object_id = repo_id.replace("/", "--")
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None

    refs_dir = os.path.join(repo_cache, "refs")
    snapshots_dir = os.path.join(repo_cache, "snapshots")
    no_exist_dir = os.path.join(repo_cache, ".no_exist")

    # Resolve refs (for instance to convert main to the associated commit sha)
    if os.path.isdir(refs_dir):
        revision_file = os.path.join(refs_dir, revision)
        if os.path.isfile(revision_file):
            with open(revision_file) as f:
                revision = f.read()

    # Check if file is cached as "no_exist"
    if os.path.isfile(os.path.join(no_exist_dir, revision, filename)):
        return _CACHED_NO_EXIST

    # Check if revision folder exists
    if not os.path.exists(snapshots_dir):
        return None
    cached_shas = os.listdir(snapshots_dir)
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    # Check if file exists in cache
    cached_file = os.path.join(snapshots_dir, revision, filename)
    return cached_file if os.path.isfile(cached_file) else None
#+end_src

##* hf_hub_download
used in utils/hub.py:270:def cached_file(

located in huggingface_hub/file_download.py

uses hf_hub_download
#+begin_src python :results none :exports code :eval no
# step 1)
url = hf_hub_url(
            repo_id,
            filename,
            subfolder=subfolder,
            repo_type=repo_type,
            revision=revision,
            endpoint=endpoint,
        )

path_or_repo_id = '/home/jup/gte-small'
filename = 'tokenizer_config.json'
subfolder= ''
repo_type= None
revision= None
cache_dir='/home/t/.cache/huggingface/hub'
user_agent="......"
force_download=None
local_files_only=True

# step 2)
cached_download(
            url,
            library_name=library_name,
            library_version=library_version,
            cache_dir=cache_dir,
            user_agent=user_agent,
            force_download=force_download,
            force_filename=force_filename,
            proxies=proxies,
            etag_timeout=etag_timeout,
            token=token,
            local_files_only=local_files_only,
            legacy_cache_layout=legacy_cache_layout,
        )
  - cache_dir = HF_HUB_CACHE  # default_cache_path
  - filename = force_filename if force_filename is not None else url_to_filename(url, etag)
  - cache_path = os.path.join(cache_dir, filename)
  - return cache_path
#+end_src

*We replace hf_hub_download with:*

#+begin_src python :results none :exports code :eval no
DEFAULT_ETAG_TIMEOUT = 10
from typing import Literal

def hf_hub_download(
        repo_id: str,
        filename: str,
        ,*,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        local_dir: Union[str, Path, None] = None,
        user_agent: Union[Dict, str, None] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        etag_timeout: float = DEFAULT_ETAG_TIMEOUT,
        token: Union[bool, str, None] = None,
        local_files_only: bool = False,
        headers: Optional[Dict[str, str]] = None,
        endpoint: Optional[str] = None,
        # Deprecated args
        legacy_cache_layout: bool = False,
        resume_download: Optional[bool] = None,
        force_filename: Optional[str] = None,
        local_dir_use_symlinks: Union[bool, Literal["auto"]] = "auto"):
    if subfolder is not None or subfolder == "":
        filename = os.path.join(subfolder, filename)
    if repo_id is None:
        repo_id = ""
    if os.path.isdir(repo_id):
        return os.path.join(repo_id, filename)
    else:
        return os.path.join(default_cache_path, filename)
#+end_src

##*     _is_offline_mode = huggingface_hub.constants.HF_HUB_OFFLINE
_is_offline_mode = True
##* name '_deprecate_method' is not defined
comment line:
: @_deprecate_method(version="4.39.0", message="This method is outdated and does not support the new cache system.")
##* NameError: name 'resolve_trust_remote_code' is not defined
models/auto/tokenization_auto.py", line 871 in from_pretrained

comment:
: trust_remote_code = resolve_trust_remote_code(
:             trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
:         )

--------------
models/auto/configuration_auto.py", line 990, in from_pretrained

-------------
auto/auto_factory.py", line 542, in from_pretrained
comment:
: trust_remote_code = resolve_trust_remote_code(
:             trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
:         )

##* safetensors_conversion.py", line 8, in <module>
: from .utils import cached_file
: # http_user_agent,
: from .utils import logging

## NameError: name 'is_remote_url' is not defined - configuration_utils.py", line 616
in utils/__init__.py
- from .hub import (is_remote_url)

in
- from .utils import (is_remote_url)

## ImportError: BertTokenizerFast requires the 🤗 Tokenizers library but it was not found in your environment. You can install it with:
: pip install tokenizers

add ~use_fast=False~ to ~AutoTokenizer.from_pretrained(~
## ImportError: cannot import name 'CLOUDFRONT_DISTRIB_PREFIX'
file_utils.py", line 26

comment lines:
- CLOUDFRONT_DISTRIB_PREFIX,
- HF_MODULES_CACHE
- HUGGINGFACE_CO_PREFIX,
- HUGGINGFACE_CO_RESOLVE_ENDPOINT,
- PYTORCH_PRETRAINED_BERT_CACHE
- PYTORCH_TRANSFORMERS_CACHE,
- S3_BUCKET_PREFIX,
- TRANSFORMERS_CACHE
- TRANSFORMERS_DYNAMIC_MODULE_NAME
- EntryNotFoundError,
- PushToHubMixin
- RepositoryNotFoundError,
- RevisionNotFoundError,
- define_sagemaker_information,
- get_cached_models
- get_file_from_repo
- has_file
- http_user_agent
## ModuleNotFoundError: No module named 'safetensors'
pytorch_utils.py", line 21, used in “id_tensor_storage()” function
- from safetensors.torch import storage_ptr, storage_size

Solution: ensure that safetensors is installed

## Disable downloading from Pypi duing installation
comment line in setup.py:
- install_requires=list(install_requires),
