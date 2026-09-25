# Copyright 2026 Google LLC
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

import os

import google.auth

_USE_VERTEXAI_ENV_VAR = "GOOGLE_GENAI_USE_VERTEXAI"


def _use_vertexai_explicitly_disabled() -> bool:
    value = os.getenv(_USE_VERTEXAI_ENV_VAR)
    return value is not None and value.strip().lower() in ("0", "false", "no")


# Only resolve Application Default Credentials when the ML Dev (API key)
# backend hasn't been explicitly selected via .env. Calling
# google.auth.default() unconditionally breaks anyone running with
# GOOGLE_GENAI_USE_VERTEXAI=0 and no gcloud ADC configured, e.g. when
# testing ADK's bidi-streaming/Live API locally against AI Studio.
if not _use_vertexai_explicitly_disabled():
    _, project_id = google.auth.default()
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    os.environ.setdefault(_USE_VERTEXAI_ENV_VAR, "True")

MODEL = os.getenv("GOOGLE_GENAI_MODEL")
if not MODEL:
    MODEL = "gemini-2.5-flash"

from . import agent  # noqa: E402
