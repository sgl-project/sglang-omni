// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "civetweb.h"

#ifdef __cplusplus
extern "C" {
#endif

int omni_connection_closed(const struct mg_connection *connection);

#ifdef __cplusplus
}
#endif
