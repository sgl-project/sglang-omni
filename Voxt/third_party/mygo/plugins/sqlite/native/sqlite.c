// A small C ABI beside SQLite. All calls from Go use integer registers,
// including doubles on Windows ARM64. Callbacks and cancellation state
// belong to C, so SQLite never retains a Go pointer or a purego callback.
#include "sqlite3.h"
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>

#ifdef _WIN32
#define MYGO_API __declspec(dllexport)
#else
#define MYGO_API __attribute__((visibility("default")))
#endif

typedef struct {
    atomic_int cancelled;
    int timeout_ms;
    int waited_ms;
    int allow_control;
} mygo_sqlite_operation;

static int authorize(void *, int, const char *, const char *, const char *, const char *);

MYGO_API int mygo_sqlite_abi_version(void) { return 1; }

MYGO_API mygo_sqlite_operation *mygo_sqlite_operation_new(int timeout_ms) {
    mygo_sqlite_operation *op = sqlite3_malloc64(sizeof(*op));
    if (op) {
        atomic_init(&op->cancelled, 0);
        op->timeout_ms = timeout_ms;
        op->waited_ms = 0;
        op->allow_control = 1;
    }
    return op;
}

MYGO_API void mygo_sqlite_operation_cancel(mygo_sqlite_operation *op) {
    atomic_store_explicit(&op->cancelled, 1, memory_order_relaxed);
}

static int progress(void *data) {
    mygo_sqlite_operation *op = data;
    return atomic_load_explicit(&op->cancelled, memory_order_relaxed);
}

static int busy(void *data, int previous) {
    mygo_sqlite_operation *op = data;
    if (!previous) op->waited_ms = 0;
    if (progress(op) || op->waited_ms >= op->timeout_ms) return 0;
    int delay = previous < 10 ? previous + 1 : 10;
    if (delay > op->timeout_ms - op->waited_ms) delay = op->timeout_ms - op->waited_ms;
    op->waited_ms += sqlite3_sleep(delay);
    return !progress(op);
}

MYGO_API void mygo_sqlite_operation_start(sqlite3 *db, mygo_sqlite_operation *op) {
    sqlite3_set_authorizer(db, authorize, op);
    sqlite3_progress_handler(db, 1000, progress, op);
    sqlite3_busy_handler(db, busy, op);
}

MYGO_API void mygo_sqlite_operation_end(sqlite3 *db) {
    sqlite3_set_authorizer(db, 0, 0);
    sqlite3_progress_handler(db, 0, 0, 0);
    sqlite3_busy_handler(db, 0, 0);
}

// Transactions belong to the batch API. ATTACH (including VACUUM INTO)
// could escape the plugin's database directory. Extension loading is
// omitted at compile time, and directory-changing pragmas are deprecated.
static int authorize(void *data, int action, const char *a, const char *b,
                     const char *database, const char *trigger) {
    mygo_sqlite_operation *op = data;
    (void)a; (void)b; (void)database; (void)trigger;
    switch (action) {
    case SQLITE_TRANSACTION:
    case SQLITE_SAVEPOINT:
        return op->allow_control ? SQLITE_OK : SQLITE_DENY;
    case SQLITE_ATTACH:
        // Ordinary VACUUM attaches an unnamed temporary database while
        // stepping. User ATTACH is denied while preparing, and VACUUM
        // INTO a named output file is denied here too.
        return op->allow_control && a && !a[0] ? SQLITE_OK : SQLITE_DENY;
    case SQLITE_DETACH:
        return op->allow_control ? SQLITE_OK : SQLITE_DENY;
    default:
        return SQLITE_OK;
    }
}

MYGO_API int mygo_sqlite_prepare(sqlite3 *db, const char *sql, int size,
                                sqlite3_stmt **stmt, const char **tail,
                                mygo_sqlite_operation *op) {
    op->allow_control = 0;
    int result = sqlite3_prepare_v2(db, sql, size, stmt, tail);
    op->allow_control = 1;
    return result;
}

MYGO_API int mygo_sqlite_bind_double_bits(sqlite3_stmt *stmt, int index, uint64_t bits) {
    double value;
    memcpy(&value, &bits, sizeof(value));
    return sqlite3_bind_double(stmt, index, value);
}

MYGO_API uint64_t mygo_sqlite_column_double_bits(sqlite3_stmt *stmt, int index) {
    double value = sqlite3_column_double(stmt, index);
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}
