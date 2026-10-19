// SPDX-License-Identifier: Apache-2.0
// Note (Adrian Wei): the pinned CivetWeb has no public peer-state query.
#include "civetweb.c"
#include "common/civetweb_connection.h"

int omni_connection_closed(const struct mg_connection *connection) {
  int socket_error = 0;
  socklen_t error_size = sizeof(socket_error);
  if (getsockopt(connection->client.sock, SOL_SOCKET, SO_ERROR, &socket_error,
                 &error_size) != 0) {
    return errno != EINTR;
  } else if (socket_error != 0) {
    return 1;
  } else {
    char pending_byte;
    const ssize_t received = recv(connection->client.sock, &pending_byte, 1,
                                  MSG_PEEK | MSG_DONTWAIT);
    return received == 0 ||
           (received < 0 && errno != EAGAIN && errno != EWOULDBLOCK &&
            errno != EINTR);
  }
}
