IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]
OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
# from huggingface_hub.constants
import os
import re
HF_HOME = os.path.expanduser(
    os.getenv(
        "HF_HOME",
        os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    )
)
HF_HUB_OFFLINE=True
default_cache_path = os.path.join(HF_HOME, "hub")
HF_HUB_CACHE = default_cache_path
HF_HUB_DISABLE_TELEMETRY=True
REGEX_COMMIT_HASH=re.compile(r"^[0-9a-f]{40}$")
