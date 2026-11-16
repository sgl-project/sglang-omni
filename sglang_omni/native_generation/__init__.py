# SPDX-License-Identifier: Apache-2.0
"""Omni support for engines that native SGLang multimodal_gen runs."""

# The native route families a generation stage serves: image generation and
# edits, video jobs with retrieval, content and deletion, and action
# generation. Kept here, free of imports, so configs can list them as the
# stage's http_routes without loading the routes.
ROUTE_PREFIXES = ("/v1/images", "/v1/videos", "/v1/actions")
