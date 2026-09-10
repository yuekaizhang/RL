# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Fix the nano-omni chat template's system-content list rendering.

vLLM's multimodal chat path normalizes string message content into parts
lists ([{"type": "text", "text": ...}]). The nano chat_template.jinja
extracts text parts for user turns but takes the SYSTEM content as-is and
pipes it through `| string`, which on a list produces the Python repr —
so every rendered prompt carried the policy document wrapped in
"[{'type': 'text', 'text': '...'}]". This patch flattens list content to
its concatenated text parts before use.

Usage:
    python yuekai_scripts/patch_nano_chat_template.py <model_dir> [...]

Each target's chat_template.jinja is backed up to chat_template.jinja.orig
(unless the backup already exists). Symlinked templates are replaced by a
regular patched file so shared HF blobs stay untouched.
"""

import shutil
import sys
from pathlib import Path

OLD = """{%- if messages[0]["role"] == "system" %}
    {%- set system_message = messages[0]["content"] %}
    {%- set loop_messages = messages[1:] %}"""

NEW = """{%- if messages[0]["role"] == "system" %}
    {%- if messages[0]["content"] is string %}
        {%- set system_message = messages[0]["content"] %}
    {%- else %}
        {%- set sysns = namespace(txt="") %}
        {%- for part in messages[0]["content"] %}
            {%- if part is mapping and "text" in part and part["text"] is string %}
                {%- set sysns.txt = sysns.txt + part["text"] %}
            {%- endif %}
        {%- endfor %}
        {%- set system_message = sysns.txt %}
    {%- endif %}
    {%- set loop_messages = messages[1:] %}"""


def patch(model_dir: str) -> None:
    tpl = Path(model_dir) / "chat_template.jinja"
    if not tpl.exists():
        print(f"SKIP (no template): {model_dir}")
        return
    text = tpl.read_text()
    if NEW.splitlines()[1].strip() in text:
        print(f"ALREADY PATCHED: {model_dir}")
        return
    if OLD not in text:
        print(f"ANCHOR NOT FOUND (template differs): {model_dir}")
        return
    backup = tpl.with_suffix(".jinja.orig")
    if not backup.exists():
        shutil.copyfile(tpl, backup)  # copyfile follows symlinks: backs up content
    patched = text.replace(OLD, NEW, 1)
    if tpl.is_symlink():
        tpl.unlink()  # break the symlink; write a regular file, keep the blob intact
    tpl.write_text(patched)
    print(f"PATCHED: {model_dir}")


if __name__ == "__main__":
    for d in sys.argv[1:]:
        patch(d)
