#include "base.h"

#if HAVE_LINUX
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>
#include <sys/sysinfo.h>
#include <dirent.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <semaphore.h>
#elif HAVE_WINDOWS
#define WIN32_LEAN_AND_MEAN
#undef min
#undef max
#include <windows.h>
#include <ShlObj_core.h>
#endif
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//////////////////////////////
// General

struct base_state {
        char *organization_name;
        char *application_name;
        enum log_level current_log_level;
        b8 is_colored_log;
        struct config_var_context *config_var_context;

        void (*arena_scratch_alloc)(void);
        void (*arena_scratch_release)(void);
        struct arena *(*scratch_arena)(i32 index);

        void *(*memory_alloc)(usize size);
        void *(*memory_realloc)(void *ptr, usize old_size, usize new_size);
        void (*memory_release)(void *ptr);
        void *(*memory_alloc_aligned)(usize size, usize alignment);
        void (*memory_release_aligned)(void *ptr);
};

static struct base_state *s_base_state;

//////////////////////////////
// Forward declarations

static void arena_scratch_alloc(void);
static void arena_scratch_release(void);
static struct arena *scratch_arena(i32 index);
static struct arena *scratch_arena_impl(i32 index);
static void arena_scratch_alloc_impl(void);
static void arena_scratch_release_impl(void);

static void *memory_alloc_impl(usize size);
static void *memory_realloc_impl(void *ptr, usize old_size, usize new_size);
static void memory_release_impl(void *ptr);
static void *memory_alloc_aligned_impl(usize size, usize alignment);
static void memory_release_aligned_impl(void *ptr);

//////////////////////////////
// General

b8 base_init(const char *org_name, const char *app_name, usize memory_size)
{
        ASSERT(!s_base_state);

        UNUSED(memory_size);

        s_base_state = memory_alloc_impl(sizeof(struct base_state));
        s_base_state->arena_scratch_alloc = arena_scratch_alloc_impl;
        s_base_state->arena_scratch_release = arena_scratch_release_impl;
        s_base_state->scratch_arena = scratch_arena_impl;

        s_base_state->memory_alloc = memory_alloc_impl;
        s_base_state->memory_realloc = memory_realloc_impl;
        s_base_state->memory_release = memory_release_impl;
        s_base_state->memory_alloc_aligned = memory_alloc_aligned_impl;
        s_base_state->memory_release_aligned = memory_release_aligned_impl;

        s_base_state->organization_name = org_name ? cstr_clone(org_name) :
                                                     NULL;
        s_base_state->application_name = app_name ? cstr_clone(app_name) : NULL;

        arena_scratch_alloc();

        return true;
}

b8 base_init_from_external(struct base_state *external_state)
{
        ASSERT(!s_base_state);
        ASSERT(external_state);
        s_base_state = external_state;
        return true;
}

void base_shutdown(void)
{
        ASSERT(s_base_state);

        memory_release(s_base_state->organization_name);
        memory_release(s_base_state->application_name);
        arena_scratch_release();
        memory_release_impl(s_base_state);
        s_base_state = NULL;
}

void base_shutdown_from_external(void)
{
        ASSERT(s_base_state);
        s_base_state = NULL;
}

struct base_state *get_base_state(void)
{
        ASSERT(s_base_state);
        return s_base_state;
}

//////////////////////////////
// Memory

static void *memory_alloc_impl(usize size)
{
        void *m = malloc(size);
        ENSURE_MSG(m, "Failed to allocate %zu bytes of memory", size);
        memory_zero(m, size);
        return m;
}

void *memory_alloc(usize size)
{
        return s_base_state->memory_alloc(size);
}

static void *memory_realloc_impl(void *ptr, usize old_size, usize new_size)
{
        void *m = realloc(ptr, new_size);
        ENSURE_MSG(m, "Failed to reallocate memory from %zu to %zu bytes",
                   old_size, new_size);
        if (new_size > old_size && m) {
                memory_zero((u8 *)m + old_size, new_size - old_size);
        }
        return m;
}

void *memory_realloc(void *ptr, usize old_size, usize new_size)
{
        return s_base_state->memory_realloc(ptr, old_size, new_size);
}

static void memory_release_impl(void *ptr)
{
        free(ptr);
}

void memory_release(void *ptr)
{
        s_base_state->memory_release(ptr);
}

static void *memory_alloc_aligned_impl(usize size, usize alignment)
{
#if HAVE_LINUX
        void *m = NULL;
        i32 result = posix_memalign(&m, alignment, size);
        if (result != 0) {
                m = NULL;
        }
#elif HAVE_WINDOWS
        void *m = _aligned_malloc(size, alignment);
#else
#error "Unsupported platform!"
#endif
        ENSURE_MSG(m,
                   "Failed to allocate %zu bytes of memory with %zu alignment",
                   size, alignment);
        memory_zero(m, size);
        return m;
}

void *memory_alloc_aligned(usize size, usize alignment)
{
        return s_base_state->memory_alloc_aligned(size, alignment);
}

static void memory_release_aligned_impl(void *ptr)
{
        free(ptr);
}

void memory_release_aligned(void *ptr)
{
        s_base_state->memory_release_aligned(ptr);
}

void *memory_set(void *ptr, usize ptr_size, u8 value, usize size)
{
        usize s = ptr_size < size ? ptr_size : size;
        return memset(ptr, value, s);
}

void *memory_zero(void *ptr, usize size)
{
        return memory_set(ptr, size, 0, size);
}

void *memory_copy(void *dest, usize dest_size, const void *src, usize src_size)
{
        usize copy_size = dest_size < src_size ? dest_size : src_size;
        return memcpy(dest, src, copy_size);
}

b8 memory_equal(const void *ptr1, usize ptr1Size, void *ptr2, usize ptr2Size)
{
        usize compareSize = ptr1Size < ptr2Size ? ptr1Size : ptr2Size;
        return (b8)(memcmp(ptr1, ptr2, compareSize) == 0);
}

//////////////////////////////
// Formatting and printing

i32 format_cstr(char *buffer, usize buffer_size, const char *fmt, ...)
{
        va_list args;
        va_start(args, fmt);
        i32 r = format_cstr_va(buffer, buffer_size, fmt, args);
        va_end(args);
        return r;
}

i32 format_cstr_va(char *buffer, usize buffer_size, const char *fmt,
                   va_list args)
{
#if HAVE_STB_SPRINTF
        return stbsp_vsnprintf(buffer, (i32)buffer_size, fmt, args);
#else
        return vsnprintf(buffer, buffer_size, fmt, args);
#endif
}

#if HAVE_STB_SPRINTF
static char *format_stb_printf_cb(const char *buf, void *user, i32 len)
{
        b8 write_error = *(b8 *)user;
        if (write_error) {
                print_error_fmt("%.*s", len, buf);
        } else {
                print_fmt("%.*s", len, buf);
        }
        return (char *)buf;
}
#endif

void print_fmt(const char *fmt, ...)
{
#if HAVE_STB_SPRINTF
        b8 write_error = false;
        char buffer[STB_SPRINTF_MIN];

        va_list args;
        va_start(args, fmt);
        stbsp_vsprintfcb(format_stb_printf_cb, &write_error, buffer, fmt, args);
        va_end(args);
#else
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
#endif
}

void print_error_fmt(const char *fmt, ...)
{
#if HAVE_STB_SPRINTF
        b8 write_error = true;
        char buffer[STB_SPRINTF_MIN];

        va_list args;
        va_start(args, fmt);
        stbsp_vsprintfcb(format_stb_printf_cb, &write_error, buffer, fmt, args);
        va_end(args);
#else
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
#endif
}

//////////////////////////////
// Logging

#if HAVE_ANDROID
#include <android/log.h>
#endif

static const char *log_level_to_str(enum log_level kind)
{
        switch (kind) {
        case LOG_LEVEL_DEBUG:
                return "[DEBUG]";
        case LOG_LEVEL_INFO:
                return "[INFO]";
        case LOG_LEVEL_ERROR:
                return "[ERROR]";
        default:
                return "[DEBUG]";
        }
}

void log_set_level(enum log_level kind)
{
        ASSERT(s_base_state);
        s_base_state->current_log_level = kind;
}

void log_set_colored_output(b8 enable)
{
        s_base_state->is_colored_log = enable;
}

void log_message(enum log_level level, const char *message, u64 length)
{
        ASSERT(s_base_state);
        if (level >= s_base_state->current_log_level) {
#if HAVE_ANDROID
                switch (kind) {
                case LOG_LEVEL_DEBUG:
                        __android_log_print(ANDROID_LOG_DEBUG, "app", "%.*s",
                                            (I32)length, message);
                        break;
                case LOG_LEVEL_INFO:
                        __android_log_print(ANDROID_LOG_INFO, "app", "%.*s",
                                            (I32)length, message);
                        break;
                case LOG_LEVEL_ERROR:
                        __android_log_print(ANDROID_LOG_ERROR, "app", "%.*s",
                                            (I32)length, message);
                        break;
                }
#else
                const char *reset = "\033[0m";
                const char *color = reset;
                if (s_base_state->is_colored_log) {
                        switch (level) {
                        case LOG_LEVEL_DEBUG:
                                color = reset; // Reset, no color
                                break;
                        case LOG_LEVEL_INFO:
                                color = "\033[0;34m"; // Blue
                                break;
                        case LOG_LEVEL_ERROR:
                                color = "\033[1;31m"; // Red
                                break;
                        }
                }

                print_fmt("%s%s %.*s%s\n", color, log_level_to_str(level),
                          (i32)length, message, reset);
#endif
        }
}

void log_message_fmt(enum log_level level, const char *fmt, ...)
{
        if (level >= s_base_state->current_log_level) {
                va_list args;
                va_start(args, fmt);
                char buffer[1024];
                format_cstr_va(buffer, sizeof(buffer), fmt, args);
                va_end(args);
                log_message(level, buffer, I32_TO_USIZE(cstr_length(buffer)));
        }
}

//////////////////////////////
// Assertions

void fail_func(const char *file, int line, const char *func,
               const char *message)
{
        LOG_ERROR("Failure in %s:%d %s(): %s", file, line, func, message);
        TRAP();
}

//////////////////////////////
// Arena

struct arena *arena_alloc_from_config(struct arena_config *config)
{
        u64 reserve_size = config->reserve_size + ARENA_HEADER_SIZE;
        u64 commit_size = config->commit_size;

        struct os_system_info *system_info;
        os_get_system_info(&system_info);

        reserve_size = align_pow2(reserve_size, system_info->page_size);
        commit_size = align_pow2(commit_size, system_info->page_size);

        void *base = os_memory_reserve(reserve_size);
        os_memory_commit(base, commit_size);

        if (!base) {
                FAIL_MSG("Failed to reserve memory for arena");
        }

        struct arena *arena = (struct arena *)base;
        arena->current = arena;
        arena->committed_size = config->commit_size;
        arena->reserved_size = config->reserve_size;
        arena->base_position = 0;
        arena->position = ARENA_HEADER_SIZE;
        arena->commit = commit_size;
        arena->reserved = reserve_size;
        arena->free_size = 0;
        arena->free_last = NULL;

        asan_poison_memory_region(base, commit_size);
        asan_unpoison_memory_region(base, ARENA_HEADER_SIZE);

        return arena;
}

struct arena *arena_alloc(void)
{
        struct arena_config config = {
                .reserve_size = ARENA_DEFAULT_RESERVE_SIZE,
                .commit_size = ARENA_DEFAULT_COMMIT_SIZE,
                .flags = ARENA_FLAG_NONE,
        };
        return arena_alloc_from_config(&config);
}

void arena_release(struct arena *arena)
{
        for (struct arena *n = arena->current, *prev = NULL; n != NULL;
             n = prev) {
                prev = n->prev;
                os_memory_release(n, n->reserved);
        }
}

void *arena_push(struct arena *arena, u64 size, u64 align)
{
        ASSERT(arena);

        if (size == 0) {
                return NULL;
        }

        struct arena *current = arena->current;
        u64 position_prev = align_pow2(current->position, align);
        u64 position_post = position_prev + size;

        // Chain, if not enough space
        if (current->reserved < position_post &&
            !(arena->flags & ARENA_FLAG_NO_CHAIN)) {
                struct arena *new_block = NULL;

#ifdef ARENA_USE_FREE_LIST
                // Try to find a free block that is big enough to fit the new allocation
                struct arena *prev_block;

                for (new_block = arena->free_last, prev_block = NULL;
                     new_block != NULL;
                     prev_block = new_block, new_block = new_block->prev) {
                        if (new_block->reserved >= align_pow2(size, align)) {
                                if (prev_block) {
                                        prev_block->prev = new_block->prev;
                                } else {
                                        arena->free_last = new_block->prev;
                                }

                                arena->free_size -= new_block->reserved_size;
                                asan_unpoison_memory_region(
                                        (u8 *)new_block + ARENA_HEADER_SIZE,
                                        new_block->reserved_size -
                                                ARENA_HEADER_SIZE);
                                break;
                        }
                }
#endif

                if (!new_block) {
                        u64 reserved_size = current->reserved_size;
                        u64 commit_size = current->committed_size;

                        if (size + ARENA_HEADER_SIZE > reserved_size) {
                                reserved_size = align_pow2(
                                        size + ARENA_HEADER_SIZE, align);
                                commit_size = align_pow2(
                                        size + ARENA_HEADER_SIZE, align);
                        }
                        struct arena_config config = {
                                .reserve_size = reserved_size,
                                .commit_size = commit_size,
                                .flags = ARENA_FLAG_NONE,
                        };
                        new_block = arena_alloc_from_config(&config);
                }

                new_block->base_position =
                        current->base_position + current->reserved;
                new_block->prev = arena->current;
                arena->current = new_block;
                current = new_block;
                position_prev = align_pow2(current->position, align);
                position_post = position_prev + size;
                ASSERT(position_post <= current->reserved);
        }

        // Commit new pages, if needed
        if (current->commit < position_post) {
                u64 commit_post_aligned =
                        position_post + current->committed_size - 1;

                commit_post_aligned -=
                        commit_post_aligned % current->committed_size;
                u64 commit_post_clamped =
                        clamp_top(commit_post_aligned, current->reserved);
                u64 commit_size = commit_post_clamped - current->commit;
                u8 *commit_ptr = (u8 *)current + current->commit;

                os_memory_commit(commit_ptr, commit_size);
                current->commit = commit_post_clamped;
        }

        // Push onto current block
        void *result = NULL;

        if (current->commit >= position_post) {
                result = ((u8 *)current) + position_prev;
                current->position = position_post;
                asan_unpoison_memory_region(result, size);
        }

        // Panic on failure
        if (!result) {
                FAIL_MSG("Arena allocation failed");
        }

        return result;
}

char *arena_push_cstr(struct arena *arena, const char *cstr)
{
        ASSERT(arena);

        usize length = cstr_length(cstr);
        char *dest = arena_push_array(arena, char, length + 1);
        cstr_copy(dest, cstr, length + 1);
        return dest;
}

char *arena_push_cstr_fmt(struct arena *arena, const char *fmt, ...)
{
        ASSERT(arena);

        va_list args;
        va_start(args, fmt);
        char *result = arena_push_cstr_fmt_va(arena, fmt, args);
        va_end(args);
        return result;
}

char *arena_push_cstr_fmt_va(struct arena *arena, const char *fmt, va_list args)
{
        ASSERT(arena);

        char temp[1024];
        i32 length = format_cstr_va(temp, sizeof(temp), fmt, args);
        if (length < 0) {
                FAIL_MSG("Failed to format string");
        }
        char *dest = arena_push_array(arena, char, (u64)length + 1);
        cstr_copy(dest, temp, (u64)length + 1);
        return dest;
}

u64 arena_position(struct arena *arena)
{
        ASSERT(arena);

        struct arena *current = arena->current;
        return current->base_position + current->position;
}

void arena_pop_to(struct arena *arena, u64 position)
{
        u64 big_position = clamp_bottom(ARENA_HEADER_SIZE, position);
        struct arena *current = arena->current;

#ifdef ARENA_USE_FREE_LIST
        for (struct arena *prev = NULL; current->base_position >= big_position;
             current = prev) {
                prev = current->prev;
                current->position = ARENA_HEADER_SIZE;
                arena->free_size += current->reserved_size;
                current->prev = arena->free_last;
                arena->free_last = current;
                asan_poison_memory_region((u8 *)current + ARENA_HEADER_SIZE,
                                          current->reserved_size -
                                                  ARENA_HEADER_SIZE);
        }
#else
        for (struct arena *prev = NULL; current->base_position >= big_position;
             current = prev) {
                prev = current->prev;
                os_memory_release(current, current->reserved);
        }
#endif

        arena->current = current;
        u64 new_position = big_position - current->base_position;

        ASSERT(new_position <= current->position);
        asan_poison_memory_region((u8 *)current + new_position,
                                  (current->position - new_position));
        current->position = new_position;
}

void arena_clear(struct arena *arena)
{
        arena_pop_to(arena, 0);
}

void arena_pop(struct arena *arena, u64 size)
{
        u64 position_old = arena_position(arena);
        u64 position_new = position_old;
        if (size < position_old) {
                position_new = position_old - size;
        }
        arena_pop_to(arena, position_new);
}

void *arena_get_base(struct arena *arena)
{
        return (u8 *)arena + ARENA_HEADER_SIZE;
}

struct arena_temp arena_temp_begin(struct arena *arena)
{
        u64 position = arena_position(arena);
        struct arena_temp temp = {
                .arena = arena,
                .position = position,
        };
        return temp;
}

void arena_temp_end(struct arena_temp temp)
{
        arena_pop_to(temp.arena, temp.position);
}

static THREAD_LOCAL struct arena *s_scratch_arenas[2];

static void set_scratch_arena(i32 index, struct arena *arena)
{
        ASSERT(index >= 0 && index < (i32)ARRAY_COUNT(s_scratch_arenas));
        s_scratch_arenas[index] = arena;
}

static struct arena *scratch_arena_impl(i32 index)
{
        ASSERT(index >= 0 && index < (i32)ARRAY_COUNT(s_scratch_arenas));
        return s_scratch_arenas[index];
}

static struct arena *scratch_arena(i32 index)
{
        return s_base_state->scratch_arena(index);
}

static void arena_scratch_alloc_impl(void)
{
        for (i32 i = 0; i < ARRAY_COUNT(s_scratch_arenas); i++) {
                if (!scratch_arena(i)) {
                        set_scratch_arena(i, arena_alloc());
                }
        }
}

static void arena_scratch_alloc(void)
{
        ASSERT(s_base_state && s_base_state->arena_scratch_alloc);
        s_base_state->arena_scratch_alloc();
}

static void arena_scratch_release_impl(void)
{
        for (i32 i = 0; i < ARRAY_COUNT(s_scratch_arenas); i++) {
                if (scratch_arena(i)) {
                        arena_release(scratch_arena(i));
                        set_scratch_arena(i, NULL);
                }
        }
}

static void arena_scratch_release(void)
{
        ASSERT(s_base_state && s_base_state->arena_scratch_release);
        s_base_state->arena_scratch_release();
}

struct arena_temp arena_scratch_begin(struct arena **conflicts,
                                      i32 conflictCount)
{
        struct arena_temp scratch = { 0 };
        for (i32 i = 0; i < ARRAY_COUNT(s_scratch_arenas); i++) {
                b8 is_conflicting = false;
                for (i32 j = 0; j < conflictCount; j++) {
                        struct arena *conflict = conflicts[j];
                        if (scratch_arena(i) == conflict) {
                                is_conflicting = true;
                                break;
                        }
                }
                if (is_conflicting == 0) {
                        scratch.arena = scratch_arena(i);
                        scratch.position = scratch.arena->position;
                        break;
                }
        }
        return scratch;
}

void arena_scratch_end(struct arena_temp scratch)
{
        arena_temp_end(scratch);
}

//////////////////////////////
// C string utilities

usize cstr_length(const char *str)
{
        if (!str) {
                return 0;
        }
        return strlen(str);
}

b8 cstr_is_digit(char c)
{
        return (c >= '0' && c <= '9');
}

b8 cstr_is_alpha(char c)
{
        return ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'));
}

b8 cstr_equal(const char *a, const char *b)
{
        return strcmp(a, b) == 0;
}

b8 cstr_equaln(const char *a, const char *b, u64 n)
{
        return strncmp(a, b, n) == 0;
}

void cstr_copy(char *dest, const char *src, u64 destSize)
{
#if HAVE_WINDOWS
        strncpy_s(dest, destSize, src, _TRUNCATE);
#else
        strncpy(dest, src, destSize);
        dest[destSize - 1] = '\0';
#endif
}

u64 cstr_hash(const char *str)
{
#if HAVE_WYHASH
        return wyhash(str, cstr_length(str), 0, _wyp);
#else
        // FNV-1a hash
        const u64 fnv_prime = 1099511628211ULL;
        const u64 fnv_offset_basis = 14695981039346656037ULL;

        u64 hash = fnv_offset_basis;
        for (const unsigned char *p = (const unsigned char *)str; *p != '\0';
             p++) {
                hash ^= (u64)(*p);
                hash *= fnv_prime;
        }
        return hash;
#endif
}

char *cstr_clone(const char *str)
{
        usize length = cstr_length(str);
        char *new_str = (char *)memory_alloc(length + 1);
        cstr_copy(new_str, str, length + 1);
        return new_str;
}

char *cstr_ends_with(const char *str, const char *suffix)
{
        usize str_len = cstr_length(str);
        usize suffix_len = cstr_length(suffix);

        if (suffix_len > str_len) {
                return NULL;
        }

        const char *str_suffix = str + (str_len - suffix_len);
        if (cstr_equaln(str_suffix, suffix, suffix_len)) {
                return (char *)str_suffix;
        } else {
                return NULL;
        }
}

char *cstr_starts_with(const char *str, const char *prefix)
{
        usize prefix_len = cstr_length(prefix);

        if (cstr_equaln(str, prefix, prefix_len)) {
                return (char *)str + prefix_len;
        } else {
                return NULL;
        }
}

char *cstr_find_last_char(const char *str, char c)
{
        if (!str) {
                return NULL;
        }
        const char *p = str;
        const char *result = NULL;

        while (*p) {
                if (*p == (char)c) {
                        result = p;
                }
                p++;
        }

        // Check the terminating '\0' too, for c == '\0'
        if ((char)c == '\0') {
                result = p;
        }

        return (char *)result;
}

char *cstr_find_char(const char *str, char c)
{
        if (!str) {
                return NULL;
        }
        char ch = (char)c;

        while (*str) {
                if (*str == ch) {
                        return (char *)str;
                }
                str++;
        }

        /* Check the terminating '\0' as well */
        if (ch == '\0') {
                return (char *)str;
        }

        return NULL;
}

i32 cstr_to_i32(const char *str)
{
        return atoi(str);
}

f32 cstr_to_f32(const char *str)
{
        return F64_TO_F32(atof(str));
}

//////////////////////////////
// String View

struct string_view string_view_from_parts(const char *data, usize length)
{
        struct string_view view;
        view.str = data;
        view.length = length;
        return view;
}

struct string_view string_view_from_cstr(const char *cstr)
{
        struct string_view view;
        view.str = cstr;
        view.length = (usize)cstr_length(cstr);
        return view;
}

b8 string_view_equal(struct string_view a, struct string_view b)
{
        return (a.length == b.length) &&
               memory_equal(a.str, a.length, (void *)b.str, b.length);
}

b8 string_view_equal_cstr(struct string_view view, const char *cstr)
{
        usize cstr_len = cstr_length(cstr);
        if (view.length != cstr_len) {
                return false;
        }
        return memory_equal(view.str, view.length, (void *)cstr, cstr_len);
}
// Lexer

struct lexer lexer_init(const char *file_path, const char *source)
{
        ASSERT(source);

        struct lexer lexer = { 0 };
        lexer.filePath = file_path;
        lexer.source = source;
        lexer.source_length = cstr_length(source);
        lexer.position = (char *)source;
        lexer.line = 1;
        lexer.column = 1;

        return lexer;
}

static b8 lexer_is_end_of_line(char c)
{
        return ((c == '\n') || (c == '\r'));
}

static b8 lexer_is_whitespace(char c)
{
        return ((c == ' ') || (c == '\t') || (c == '\v') || (c == '\f') ||
                lexer_is_end_of_line(c));
}

static b8 lexer_is_alpha(char c)
{
        return (((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')));
}

static b8 lexer_is_number(char c)
{
        return ((c >= '0') && (c <= '9'));
}

static b8 lexer_next_is_not_eof(struct lexer *lexer)
{
        // Check if the next character that follows after the current position is not the end of file.
        return (lexer->source + lexer->source_length) > (lexer->position + 1);
}

static void lexer_increase_position(char **position, usize *line, usize *column)
{
        if ((*position)[0] == '\n') {
                ++(*line);
                *column = 1;
        } else {
                ++(*column);
        }
        ++(*position);
}

static void lexer_decrease_position(char **position, usize *line, usize *column)
{
        --(*position);
        if ((*position)[0] == '\n') {
                --(*line);
                *column = 1;
        } else {
                --(*column);
        }
}

static void lexer_skip_whitespace(struct lexer *lexer)
{
        // Scan text until whitespace is finished.
        while (true) {
                // Check if it is a pure whitespace first.
                if (lexer_is_whitespace(lexer->position[0])) {
                        // Handle change of line
                        // if (lexer_is_end_of_line(position[0]))
                        //     ++line_;
                        // Advance to next character
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                }
                // Check for single line # comments
                else if (lexer->comment_type == COMMENT_TYPE_HASH &&
                         lexer->position[0] == '#') {
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                        while (lexer->position[0] &&
                               !lexer_is_end_of_line(lexer->position[0])) {
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                        }
                }
                // Check for single line comments ("//")
                else if (lexer->comment_type == COMMENT_TYPE_DOUBLE_SLASH &&
                         lexer->position[0] == '/' &&
                         lexer->position[1] == '/') {
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                        while (lexer->position[0] &&
                               !lexer_is_end_of_line(lexer->position[0])) {
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                        }
                }
                // Check for c-style comments
                else if (lexer->comment_type == COMMENT_TYPE_DOUBLE_SLASH &&
                         lexer->position[0] == '/' &&
                         lexer->position[1] == '*') {
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                        // Advance until the string is closed. Remember to check if line is
                        // changed.
                        while (!(lexer->position[0] == '*' &&
                                 lexer->position[1] == '/')) {
                                // Advance to next character
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                        }
                        if (lexer->position[0] == '*') {
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                        }
                } else {
                        break;
                }
        }
}

static f64 lexer_parse_number(struct lexer *lexer)
{
        char c = lexer->position[0];

        // Parse the following literals:
        // 58, -58, 0.003, 4e2, 123.456e-67, 0.1E4f, 1.0f
        // 1. Sign detection
        i32 sign = 1;

        if (c == '-') {
                sign = -1;
                lexer_increase_position(&lexer->position, &lexer->line,
                                        &lexer->column);
        }

        // 2. Heading zeros (00.003)
        if (*lexer->position == '0') {
                lexer_increase_position(&lexer->position, &lexer->line,
                                        &lexer->column);
                while (*lexer->position == '0') {
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                }
        }

        // 3. Decimal part (until the point)
        i32 decimal_part = 0;

        if (*lexer->position > '0' && *lexer->position <= '9') {
                decimal_part = (*lexer->position - '0');
                lexer_increase_position(&lexer->position, &lexer->line,
                                        &lexer->column);
                while (*lexer->position != '.' &&
                       lexer_is_number(*lexer->position)) {
                        decimal_part =
                                (decimal_part * 10) + (*lexer->position - '0');
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                }
        }

        // 4. Fractional part
        i32 fractional_part = 0;
        i32 fractional_divisor = 1;

        if (*lexer->position == '.') {
                lexer_increase_position(&lexer->position, &lexer->line,
                                        &lexer->column);
                static const u8 max_fractional_digits = 9;
                u8 fractional_digits = 0;

                while (lexer_is_number(*lexer->position)) {
                        // Limit the number of fractional digits to avoid overflow, but still parse everything
                        if (fractional_digits < max_fractional_digits) {
                                fractional_part = (fractional_part * 10) +
                                                  (*lexer->position - '0');
                                fractional_divisor *= 10;
                        }
                        ++fractional_digits;
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                }
        }

        // 6. Literal (if present)
        if (*lexer->position == 'f' || *lexer->position == 'F') {
                lexer_increase_position(&lexer->position, &lexer->line,
                                        &lexer->column);
        }

        // 6. Exponent (if present)
        if (*lexer->position == 'e' || *lexer->position == 'E') {
                lexer_increase_position(&lexer->position, &lexer->line,
                                        &lexer->column);
        }

        f64 parsed_number = (f64)sign * (decimal_part + ((f64)fractional_part /
                                                         fractional_divisor));

        return parsed_number;
}

struct token lexer_next_token(struct lexer *lexer)
{
        if (lexer->is_revert_token) {
                lexer->is_revert_token = false;
                return lexer->last_token;
        }

        lexer_skip_whitespace(lexer);

        // Init token
        struct token token = { 0 };

        token.type = TOKEN_TYPE_UNKNOWN;
        token.text = string_view_from_parts(lexer->position, 1);
        token.line = lexer->line;
        token.column = lexer->column;

        if ((lexer->source + lexer->source_length) - lexer->position == 0) {
                token.text = string_view_from_parts(NULL, 0);
                token.type = TOKEN_TYPE_END_OF_STREAM;
                lexer->is_eof = true;
                lexer->last_token = token;
                return token;
        }

        char c = lexer->position[0];

        lexer_increase_position(&lexer->position, &lexer->line, &lexer->column);

        switch (c) {
        case '(':
                token.type = TOKEN_TYPE_OPEN_PAREN;
                break;
        case ')':
                token.type = TOKEN_TYPE_CLOSE_PAREN;
                break;
        case '[':
                token.type = TOKEN_TYPE_OPEN_BRACKET;
                break;
        case ']':
                token.type = TOKEN_TYPE_CLOSE_BRACKET;
                break;
        case '{':
                token.type = TOKEN_TYPE_OPEN_BRACE;
                break;
        case '}':
                token.type = TOKEN_TYPE_CLOSE_BRACE;
                break;
        case '<':
                token.type = TOKEN_TYPE_OPEN_ANGLE_BRACKET;
                break;
        case '>':
                token.type = TOKEN_TYPE_CLOSE_ANGLE_BRACKET;
                break;
        case ':':
                token.type = TOKEN_TYPE_COLON;
                break;
        case ';':
                token.type = TOKEN_TYPE_SEMICOLON;
                break;
        case '*':
                token.type = TOKEN_TYPE_ASTERISK;
                break;
        case '#':
                token.type = TOKEN_TYPE_HASH;
                break;
        case ',':
                token.type = TOKEN_TYPE_COMMA;
                break;
        case '=':
                token.type = TOKEN_TYPE_EQUAL;
                break;
        case '/':
                token.type = TOKEN_TYPE_DIVIDE;
                break;
        case '+':
                token.type = TOKEN_TYPE_PLUS;
                break;
        case '"':
                token.type = TOKEN_TYPE_STRING;
                while (lexer->position[0] && lexer->position[0] != '"') {
                        if (lexer->position[0] == '\\' && lexer->position[1]) {
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                        }
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                }
                token.text.str += 1; // Skip the first "
                token.text.length = (u32)(lexer->position - token.text.str);
                if (lexer->position[0] == '"') {
                        lexer_increase_position(&lexer->position, &lexer->line,
                                                &lexer->column);
                }
                break;
        default:
                // Identifier or keyword
                if (lexer_is_alpha(c)) {
                        token.type = TOKEN_TYPE_IDENTIFIER;
                        while (lexer_is_alpha(lexer->position[0]) ||
                               lexer_is_number(lexer->position[0]) ||
                               lexer->position[0] == '_') {
                                lexer_increase_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                        }
                        token.text.length =
                                (u32)(lexer->position - token.text.str);
                }
                // Numbers
                else if (lexer_is_number(c) || c == '-') {
                        if (c == '-' && !lexer_next_is_not_eof(lexer)) {
                                token.type = TOKEN_TYPE_MINUS;
                                break;
                        } else if (c == '-' && lexer_next_is_not_eof(lexer) &&
                                   !lexer_is_number(lexer->position[1]) &&
                                   lexer->position[1] != '.') {
                                token.type = TOKEN_TYPE_MINUS;
                                break;
                        } else {
                                // Backtrack to the start of the number
                                lexer_decrease_position(&lexer->position,
                                                        &lexer->line,
                                                        &lexer->column);
                                f64 number = lexer_parse_number(lexer);

                                token.type = TOKEN_TYPE_NUMBER;
                                token.number = number;
                                token.text.length =
                                        (u32)(lexer->position - token.text.str);
                        }
                } else {
                        token.type = TOKEN_TYPE_UNKNOWN;
                }
        }

        lexer->last_token = token;
        return token;
}

void lexer_revert_token(struct lexer *lexer)
{
        ASSERT(!lexer->is_revert_token);
        lexer->is_revert_token = true;
}

struct token lexer_peek_token(struct lexer *lexer)
{
        struct token token = lexer_next_token(lexer);
        lexer_revert_token(lexer);
        return token;
}

static void lexer_write_error(struct lexer *lexer, const char *message)
{
        LOG_ERROR("%s:%" PRIuz ":%" PRIuz ": error: expected %s",
                  lexer->filePath, lexer->line, lexer->column, message);
}

b8 lexer_expect_identifier(struct lexer *lexer, const char *identifier,
                           struct token *out_token, b8 fatal)
{
        struct token token = lexer_next_token(lexer);
        if (out_token) {
                *out_token = token;
        }
        b8 success = token.type == TOKEN_TYPE_IDENTIFIER &&
                     string_view_equal_cstr(token.text, identifier);
        if (fatal && !success) {
                lexer_write_error(lexer, "identifier");
        }
        return success;
}

static const char *token_type_to_string(enum token_type type)
{
        switch (type) {
        case TOKEN_TYPE_UNKNOWN:
                return "unknown";
        case TOKEN_TYPE_OPEN_PAREN:
                return "(";
        case TOKEN_TYPE_CLOSE_PAREN:
                return ")";
        case TOKEN_TYPE_OPEN_BRACKET:
                return "[";
        case TOKEN_TYPE_CLOSE_BRACKET:
                return "]";
        case TOKEN_TYPE_OPEN_BRACE:
                return "{";
        case TOKEN_TYPE_CLOSE_BRACE:
                return "}";
        case TOKEN_TYPE_OPEN_ANGLE_BRACKET:
                return "<";
        case TOKEN_TYPE_CLOSE_ANGLE_BRACKET:
                return ">";
        case TOKEN_TYPE_COLON:
                return ":";
        case TOKEN_TYPE_SEMICOLON:
                return ";";
        case TOKEN_TYPE_ASTERISK:
                return "*";
        case TOKEN_TYPE_EQUAL:
                return "=";
        case TOKEN_TYPE_PLUS:
                return "+";
        case TOKEN_TYPE_MINUS:
                return "-";
        case TOKEN_TYPE_DIVIDE:
                return "/";
        case TOKEN_TYPE_HASH:
                return "#";
        case TOKEN_TYPE_COMMA:
                return ",";
        case TOKEN_TYPE_STRING:
                return "string";
        case TOKEN_TYPE_IDENTIFIER:
                return "identifier";
        case TOKEN_TYPE_NUMBER:
                return "number";
        case TOKEN_TYPE_END_OF_STREAM:
                return "end of stream";
                break;
        }
        return "unknown";
}

b8 lexer_expect_token_kind(struct lexer *lexer, enum token_type type,
                           struct token *out_token, b8 fatal)
{
        struct token token = lexer_next_token(lexer);
        if (out_token) {
                *out_token = token;
        }
        b8 success = token.type == type;
        if (fatal && !success) {
                lexer_write_error(lexer, token_type_to_string(type));
        }
        return success;
}

b8 lexer_expect_string(struct lexer *lexer, const char *str,
                       struct token *out_token, b8 fatal)
{
        struct token token = lexer_next_token(lexer);
        if (out_token) {
                *out_token = token;
        }
        b8 success = token.type == TOKEN_TYPE_STRING &&
                     string_view_equal_cstr(token.text, str);
        if (fatal && !success) {
                lexer_write_error(lexer, "string");
        }
        return success;
}

b8 lexer_expect_number(struct lexer *lexer, f64 number, struct token *out_token,
                       b8 fatal)
{
        struct token token = lexer_next_token(lexer);
        if (out_token) {
                *out_token = token;
        }
        b8 success = token.type == TOKEN_TYPE_NUMBER && token.number == number;

        if (fatal && !success) {
                lexer_write_error(lexer, "number");
        }

        return success;
}

//////////////////////////////
// SD file format parser

struct sd_parser {
        jmp_buf jmp;
        struct arena *arena;
        struct lexer lexer;
        struct sd_node *root;
        b8 is_error;
};

static void sd_parse_error(struct sd_parser *parser)
{
        parser->is_error = true;
        longjmp(parser->jmp, 1);
}

static void sd_parse_error_msg(struct sd_parser *parser, const char *message)
{
        LOG_ERROR("%s:%" PRIuz ":%" PRIuz ": error: %s", parser->lexer.filePath,
                  parser->lexer.line, parser->lexer.column, message);
        parser->is_error = true;
        longjmp(parser->jmp, 1);
}

static struct sd_node *sd_node_alloc(struct arena *arena,
                                     enum sd_node_type type)
{
        struct sd_node *node = arena_push_array(arena, struct sd_node, 1);
        node->type = type;
        return node;
}

static void sd_node_append_child(struct sd_node *parent, struct sd_node *child)
{
        if (!parent->first_child) {
                parent->first_child = child;
                parent->last_child = child;
        } else {
                parent->last_child->next = child;
                parent->last_child = child;
        }
        parent->child_count++;
}

static struct sd_node *sd_parse_object(struct sd_parser *parser,
                                       b8 skip_braces);
static struct sd_node *sd_parse_array(struct sd_parser *parser,
                                      b8 skip_brackets);

static struct sd_node *sd_parse_value(struct sd_parser *parser)
{
        struct token token = lexer_next_token(&parser->lexer);

        switch (token.type) {
        case TOKEN_TYPE_NUMBER: {
                struct sd_node *node =
                        sd_node_alloc(parser->arena, SD_NODE_TYPE_NUMBER);
                node->value.number = token.number;
                return node;
        }

        case TOKEN_TYPE_STRING: {
                struct sd_node *node =
                        sd_node_alloc(parser->arena, SD_NODE_TYPE_STRING);
                node->value.str = token.text;
                return node;
        }

        case TOKEN_TYPE_IDENTIFIER: {
                struct sd_node *node =
                        sd_node_alloc(parser->arena, SD_NODE_TYPE_BOOLEAN);
                if (string_view_equal_cstr(token.text, "true")) {
                        node->value.boolean = true;
                        return node;
                } else if (string_view_equal_cstr(token.text, "false")) {
                        node->value.boolean = false;
                        return node;
                } else {
                        sd_parse_error_msg(parser, "Expected true or false");
                        return NULL;
                }
        }

        case TOKEN_TYPE_OPEN_BRACE: {
                lexer_revert_token(&parser->lexer);
                return sd_parse_object(parser, false);
        }

        case TOKEN_TYPE_OPEN_BRACKET: {
                lexer_revert_token(&parser->lexer);
                return sd_parse_array(parser, false);
        }

        default: {
                sd_parse_error_msg(
                        parser,
                        "Expected number, string, boolean, object, or array");
                return NULL;
        }
        }
}

static struct sd_node *sd_parse_array(struct sd_parser *parser,
                                      b8 skip_brackets)
{
        if (!skip_brackets) {
                if (!lexer_expect_token_kind(&parser->lexer,
                                             TOKEN_TYPE_OPEN_BRACKET, NULL,
                                             true)) {
                        sd_parse_error(parser);
                }
        }

        struct sd_node *array =
                sd_node_alloc(parser->arena, SD_NODE_TYPE_ARRAY);

        struct token token = { 0 };

        // Check for empty array
        token = lexer_peek_token(&parser->lexer);
        if (token.type == TOKEN_TYPE_CLOSE_BRACKET) {
                lexer_next_token(&parser->lexer); // consume ']'
                return array;
        }

        // Parse array elements
        while (true) {
                struct sd_node *element = sd_parse_value(parser);
                sd_node_append_child(array, element);

                // Check for end
                token = lexer_peek_token(&parser->lexer);
                if (token.type == (skip_brackets ? TOKEN_TYPE_END_OF_STREAM :
                                                   TOKEN_TYPE_CLOSE_BRACKET)) {
                        lexer_next_token(&parser->lexer); // consume ']' or EOS
                        return array;
                }
        }

        return array;
}

static struct sd_node *sd_parse_object(struct sd_parser *parser, b8 skip_braces)
{
        if (!skip_braces) {
                if (!lexer_expect_token_kind(&parser->lexer,
                                             TOKEN_TYPE_OPEN_BRACE, NULL,
                                             true)) {
                        sd_parse_error(parser);
                }
        }

        struct sd_node *object =
                sd_node_alloc(parser->arena, SD_NODE_TYPE_OBJECT);

        struct token token = { 0 };

        // Check for empty object
        token = lexer_peek_token(&parser->lexer);
        if (token.type == TOKEN_TYPE_CLOSE_BRACE) {
                lexer_next_token(&parser->lexer); // consume '}'
                return object;
        }

        // Parse object members
        while (true) {
                // Get the key (name)
                if (!lexer_expect_token_kind(&parser->lexer,
                                             TOKEN_TYPE_IDENTIFIER, &token,
                                             true)) {
                        sd_parse_error(parser);
                }

                struct string_view name = token.text;

                // Expect ':'
                if (!lexer_expect_token_kind(&parser->lexer, TOKEN_TYPE_COLON,
                                             NULL, true)) {
                        sd_parse_error(parser);
                }

                // Parse the value
                struct sd_node *value = sd_parse_value(parser);
                value->name = name; // Set the name for this node

                sd_node_append_child(object, value);

                // Check for end
                token = lexer_peek_token(&parser->lexer);
                if (token.type == (skip_braces ? TOKEN_TYPE_END_OF_STREAM :
                                                 TOKEN_TYPE_CLOSE_BRACE)) {
                        lexer_next_token(&parser->lexer); // consume '}' or EOS
                        return object;
                }
        }

        return object;
}

static void sd_parse_root(struct sd_parser *parser)
{
        struct token token = lexer_peek_token(&parser->lexer);
        if (token.type == TOKEN_TYPE_OPEN_BRACE) {
                parser->root = sd_parse_object(parser, false);
        } else if (token.type == TOKEN_TYPE_OPEN_BRACKET) {
                parser->root = sd_parse_array(parser, false);
        } else if (token.type == TOKEN_TYPE_IDENTIFIER) {
                // Assume root is an object without braces
                parser->root = sd_parse_object(parser, true);
        } else {
                // Assume its an array without brackets
                parser->root = sd_parse_array(parser, true);
        }

        // Ensure we've consumed all tokens
        if (!lexer_expect_token_kind(&parser->lexer, TOKEN_TYPE_END_OF_STREAM,
                                     NULL, true)) {
                sd_parse_error(parser);
        }
}

struct sd_parser *sd_parse(struct arena *arena, const char *file_path,
                           const char *source)
{
        struct sd_parser *parser = arena_push_array(arena, struct sd_parser, 1);
        parser->arena = arena;

        parser->lexer = lexer_init(file_path, source);

        // Parse the source
        if (setjmp(parser->jmp) == 0) {
                sd_parse_root(parser);
        } else {
                // If we reach here, it means we encountered an error
                parser->is_error = true;
        }

        return parser;
}

b8 sd_parser_has_errors(struct sd_parser *parser)
{
        return !parser || parser->is_error;
}

struct sd_node *sd_parser_root(struct sd_parser *parser)
{
        return parser ? parser->root : NULL;
}

struct sd_node *sd_node_find(struct sd_node *node, const char *name)
{
        if (!node || !name || name[0] == '\0') {
                return NULL;
        }

        // Search children first
        for (struct sd_node *child = node->first_child; child;
             child = child->next) {
                if (string_view_equal_cstr(child->name, name)) {
                        return child;
                }
        }

        // Then search siblings
        for (struct sd_node *sibling = node->next; sibling;
             sibling = sibling->next) {
                if (string_view_equal_cstr(sibling->name, name)) {
                        return sibling;
                }
        }

        // Didn't found anything in the first level of the childs and siblings. Recursively search in children.
        for (struct sd_node *child = node->first_child; child;
             child = child->next) {
                struct sd_node *found = sd_node_find(child, name);
                if (found) {
                        return found;
                }
        }

        return NULL;
}

b8 sd_node_find_object(struct sd_node *node, const char *name,
                       struct sd_node **outObject)
{
        if (!node || !name || name[0] == '\0') {
                *outObject = NULL;
                return false;
        }

        struct sd_node *found = sd_node_find(node, name);
        if (found && sd_node_is_object(found)) {
                *outObject = found;
                return true;
        } else {
                *outObject = NULL;
                return false;
        }
}

b8 sd_node_find_array(struct sd_node *node, const char *name,
                      struct sd_node **outArray)
{
        if (!node || !name || name[0] == '\0') {
                *outArray = NULL;
                return false;
        }

        struct sd_node *found = sd_node_find(node, name);
        if (found && sd_node_is_array(found)) {
                *outArray = found;
                return true;
        } else {
                *outArray = NULL;
                return false;
        }
}

b8 sd_node_find_string(struct sd_node *node, const char *name,
                       const char *defaultValue, struct string_view *outValue)
{
        if (!node || !name || name[0] == '\0') {
                *outValue = string_view_from_cstr(defaultValue);
                return false;
        }

        struct sd_node *found = sd_node_find(node, name);
        if (found && sd_node_is_string(found)) {
                *outValue = found->value.str;
                return true;
        } else {
                *outValue = string_view_from_cstr(defaultValue);
                return false;
        }
}

b8 sd_node_find_number(struct sd_node *node, const char *name, f64 defaultValue,
                       f64 *outValue)
{
        if (!node || !name || name[0] == '\0') {
                *outValue = defaultValue;
                return false;
        }

        struct sd_node *found = sd_node_find(node, name);
        if (found && sd_node_is_number(found)) {
                *outValue = found->value.number;
                return true;
        } else {
                *outValue = defaultValue;
                return false;
        }
}

b8 sd_node_find_boolean(struct sd_node *node, const char *name, b8 defaultValue,
                        b8 *outValue)
{
        if (!node || !name || name[0] == '\0') {
                *outValue = defaultValue;
                return false;
        }

        struct sd_node *found = sd_node_find(node, name);
        if (found && sd_node_is_boolean(found)) {
                *outValue = found->value.boolean;
                return true;
        } else {
                *outValue = defaultValue;
                return false;
        }
}

b8 sd_node_is_object(struct sd_node *node)
{
        return node && node->type == SD_NODE_TYPE_OBJECT;
}

b8 sd_node_is_array(struct sd_node *node)
{
        return node && node->type == SD_NODE_TYPE_ARRAY;
}

b8 sd_node_is_string(struct sd_node *node)
{
        return node && node->type == SD_NODE_TYPE_STRING;
}

b8 sd_node_is_number(struct sd_node *node)
{
        return node && node->type == SD_NODE_TYPE_NUMBER;
}

b8 sd_node_is_boolean(struct sd_node *node)
{
        return node && node->type == SD_NODE_TYPE_BOOLEAN;
}

b8 sd_node_object(struct sd_node *node, struct sd_node **outObject)
{
        if (sd_node_is_object(node)) {
                *outObject = node;
                return true;
        } else {
                *outObject = NULL;
                return false;
        }
}

b8 sd_node_array(struct sd_node *node, struct sd_node **outArray)
{
        if (sd_node_is_array(node)) {
                *outArray = node;
                return true;
        } else {
                *outArray = NULL;
                return false;
        }
}

b8 sd_node_string(struct sd_node *node, const char *defaultValue,
                  struct string_view *outValue)
{
        if (sd_node_is_string(node)) {
                *outValue = node->value.str;
                return true;
        } else {
                *outValue = string_view_from_cstr(defaultValue);
                return false;
        }
}

b8 sd_node_number(struct sd_node *node, f64 defaultValue, f64 *outValue)
{
        if (sd_node_is_number(node)) {
                *outValue = node->value.number;
                return true;
        } else {
                *outValue = defaultValue;
                return false;
        }
}

b8 sd_node_boolean(struct sd_node *node, b8 defaultValue, b8 *outValue)
{
        if (sd_node_is_boolean(node)) {
                *outValue = node->value.boolean;
                return true;
        } else {
                *outValue = defaultValue;
                return false;
        }
}

//////////////////////////////
// OS

void os_get_system_info(struct os_system_info **outInfo)
{
        static struct os_system_info info = { 0 };
        static b8 is_cached = false;
        if (!is_cached) {
#if HAVE_ANDROID
                info.logical_processor_count = sysconf(_SC_NPROCESSORS_ONLN);
                info.page_size = (u64)getpagesize();
                info.large_page_size = mb(2);
                info.allocation_granularity = info.page_size;
#elif HAVE_LINUX
                info.logical_processor_count = (u32)get_nprocs();
                info.page_size = (u64)getpagesize();
                info.large_page_size = mb(2);
                info.allocation_granularity = info.page_size;
#elif HAVE_WINDOWS
                SYSTEM_INFO sys_info = { 0 };
                GetSystemInfo(&sys_info);
                info.logical_processor_count = sys_info.dwNumberOfProcessors;
                info.page_size = sys_info.dwPageSize;
                info.large_page_size = GetLargePageMinimum();
                info.allocation_granularity = sys_info.dwAllocationGranularity;
#else
#error "OS not supported"
#endif
                is_cached = true;
        }

        *outInfo = &info;
}

void *os_memory_reserve(usize size)
{
#if HAVE_LINUX
        void *result =
                mmap(0, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (result == MAP_FAILED) {
                result = 0;
        }
        return result;
#elif HAVE_WINDOWS
        return VirtualAlloc(0, size, MEM_RESERVE, PAGE_READWRITE);
#else
#error "Unsupported platform"
#endif
}

void os_memory_release(void *ptr, usize size)
{
#if HAVE_LINUX
        munmap(ptr, size);
#elif HAVE_WINDOWS
        UNUSED(size);
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
#error "Unsupported platform"
#endif
}

void os_memory_commit(void *ptr, usize size)
{
#if HAVE_LINUX
        i32 res = mprotect(ptr, size, PROT_READ | PROT_WRITE);
        if (res < 0) {
                FAIL_MSG("mprotect failed to commit memory: %s",
                         strerror(errno));
        }
#elif HAVE_WINDOWS
        VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
#else
#error "Unsupported platform"
#endif
}

void os_memory_decommit(void *ptr, usize size)
{
#if HAVE_LINUX
        madvise(ptr, size, MADV_DONTNEED);
        mprotect(ptr, size, PROT_NONE);
#elif HAVE_WINDOWS
        VirtualFree(ptr, size, MEM_DECOMMIT);
#else
#error "Unsupported platform"
#endif
}

// Internal file helpers (static)

static b8 os_file_open_from_cstr(os_access_flags flags, const char *path,
                                 os_file_handle *out_file)
{
#if HAVE_LINUX
        i32 file_flags = 0;
        if (flags & OS_ACCESS_FLAG_READ && flags & OS_ACCESS_FLAG_WRITE) {
                file_flags = O_RDWR;
        } else if (flags & OS_ACCESS_FLAG_WRITE) {
                file_flags = O_WRONLY | O_TRUNC;
        } else if (flags & OS_ACCESS_FLAG_READ) {
                file_flags = O_RDONLY;
        }
        if (flags & OS_ACCESS_FLAG_APPEND) {
                file_flags |= O_APPEND;
        }
        if (flags & (OS_ACCESS_FLAG_WRITE | OS_ACCESS_FLAG_APPEND)) {
                file_flags |= O_CREAT;
        }

        i32 fd = open(path, file_flags, 0755);
        if (fd < 0) {
                return false;
        }

        *out_file = (u64)fd;

        return true;
#elif HAVE_WINDOWS
        DWORD desired_access = 0;
        if (flags & OS_ACCESS_FLAG_READ) {
                desired_access |= GENERIC_READ;
        }
        if (flags & OS_ACCESS_FLAG_WRITE) {
                desired_access |= GENERIC_WRITE;
        }
        DWORD creation_disposition = OPEN_EXISTING;
        if (flags & OS_ACCESS_FLAG_WRITE) {
                creation_disposition = CREATE_ALWAYS;
        }
        HANDLE handle = CreateFileA(path, desired_access,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                                    creation_disposition, FILE_ATTRIBUTE_NORMAL,
                                    NULL);
        if (handle == INVALID_HANDLE_VALUE) {
                return false;
        }
        *out_file = (u64)(uintptr_t)handle;
        return true;
#else
#error "Unsupported platform"
#endif
}

static void os_file_close(os_file_handle file)
{
#if HAVE_LINUX
        i32 fd = (i32)file;
        close(fd);
#elif HAVE_WINDOWS
        HANDLE handle = (HANDLE)(uintptr_t)file;
        CloseHandle(handle);
#else
#error "Unsupported platform"
#endif
}

static usize os_file_read(os_file_handle file, usize begin, usize end,
                          void *out_data)
{
        ASSERT(begin <= end);

#if HAVE_LINUX
        i32 fd = (i32)file;

        u64 bytes_to_read = end - begin;
        u64 bytes_read = 0;
        u64 bytes_left_to_read = bytes_to_read;

        for (; bytes_left_to_read > 0;) {
                usize read_result = ISIZE_TO_USIZE(pread(
                        fd, (u8 *)out_data + bytes_read, bytes_left_to_read,
                        USIZE_TO_ISIZE(begin + bytes_read)));
                if (read_result >= 0) {
                        bytes_read += read_result;
                        bytes_left_to_read -= read_result;
                } else if (errno != EINTR) {
                        break;
                }
        }
        return bytes_read;
#elif HAVE_WINDOWS
        HANDLE handle = (HANDLE)(uintptr_t)file;
        u64 bytes_to_read = end - begin;
        u64 bytes_read = 0;
        u64 bytes_left_to_read = bytes_to_read;

        LARGE_INTEGER li = { 0 };
        li.QuadPart = (LONGLONG)begin;
        SetFilePointerEx(handle, li, NULL, FILE_BEGIN);

        while (bytes_left_to_read > 0) {
                DWORD read_result = 0;
                if (ReadFile(handle, (u8 *)out_data + bytes_read,
                             (DWORD)bytes_left_to_read, &read_result, NULL)) {
                        bytes_read += read_result;
                        bytes_left_to_read -= read_result;
                        if (read_result == 0) {
                                break; // EOF
                        }
                } else {
                        break;
                }
        }
        return bytes_read;
#else
#error "Unsupported platform"
#endif
}

static b8 os_file_get_stats(os_file_handle file,
                            struct os_file_stats *out_stats)
{
#if HAVE_LINUX
        struct stat file_stat = { 0 };
        if (fstat((i32)file, &file_stat) != 0) {
                return false;
        }
        out_stats->size = (u64)file_stat.st_size;
        out_stats->last_modified_time = (u64)file_stat.st_mtime;
        return true;
#elif HAVE_WINDOWS
        HANDLE handle = (HANDLE)(uintptr_t)file;

        BY_HANDLE_FILE_INFORMATION file_info = { 0 };
        if (!GetFileInformationByHandle(handle, &file_info)) {
                return false;
        }
        out_stats->size = ((u64)file_info.nFileSizeHigh << 32) |
                          (u64)file_info.nFileSizeLow;

        FILETIME ft = file_info.ftLastWriteTime;
        ULARGE_INTEGER ull = { 0 };
        ull.LowPart = ft.dwLowDateTime;
        ull.HighPart = ft.dwHighDateTime;
        // Convert to Unix epoch time
        out_stats->last_modified_time =
                (ull.QuadPart - 116444736000000000ULL) / 10000000ULL;
        return true;
#else
#error "Unsupported platform"
#endif
}

b8 os_file_read_all_cstr(const char *path, struct arena *arena, void **out_data,
                         u64 *out_size)
{
        os_file_handle file = 0;
        if (!os_file_open_from_cstr(OS_ACCESS_FLAG_READ, path, &file)) {
                *out_data = NULL;
                return false;
        }

        struct os_file_stats stats = { 0 };
        if (!os_file_get_stats(file, &stats)) {
                *out_data = NULL;
                os_file_close(file);
                return false;
        }

        u8 *data = arena_push_array(arena, u8, stats.size);
        u64 bytes_read = os_file_read(file, 0, stats.size, data);
        os_file_close(file);

        if (bytes_read != stats.size) {
                *out_data = NULL;
                return false;
        }

        *out_data = data;
        *out_size = stats.size;

        return true;
}

b8 os_file_read_all_string_cstr(const char *path, struct arena *arena,
                                char **out_string)
{
        os_file_handle file = 0;
        if (!os_file_open_from_cstr(OS_ACCESS_FLAG_READ, path, &file)) {
                *out_string = NULL;
                return false;
        }

        struct os_file_stats stats = { 0 };
        if (!os_file_get_stats(file, &stats)) {
                *out_string = NULL;
                os_file_close(file);
                return false;
        }

        u8 *data = arena_push_array(arena, u8, stats.size + 1);
        u64 bytes_read = os_file_read(file, 0, stats.size, data);
        os_file_close(file);

        if (bytes_read != stats.size) {
                *out_string = NULL;
                return false;
        }

        data[stats.size] = '\0';
        *out_string = (char *)data;

        return true;
}

u64 os_time_now_micros(void)
{
#if HAVE_LINUX
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        return (u64)ts.tv_sec * 1000000ULL + (u64)(ts.tv_nsec / 1000LL);
#elif HAVE_WINDOWS
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft);
        u64 time = (((u64)ft.dwHighDateTime << 32) | (u64)ft.dwLowDateTime);
        // Convert from 100-nanosecond intervals since January 1, 1601 to microseconds since
        // January 1, 1970
        time -= 116444736000000000ULL;
        return time / 10ULL;
#else
#error "Unsupported platform"
#endif
}

f64 os_time_now_seconds(void)
{
        return (f64)os_time_now_micros() / 1000000.0;
}

void os_timer_start(struct os_timer *timer)
{
        timer->start_time = os_time_now_micros();
        timer->last_tick_time = timer->start_time;
}

f64 os_timer_elapsed_seconds(struct os_timer *timer)
{
        u64 now = os_time_now_micros();
        return (f64)(now - timer->start_time) / 1000000.0;
}

u64 os_timer_elapsed_micros(struct os_timer *timer)
{
        u64 now = os_time_now_micros();
        return now - timer->start_time;
}

void os_timer_tick(struct os_timer *timer)
{
        timer->last_tick_time = os_time_now_micros();
}

f64 os_timer_delta_seconds(struct os_timer *timer)
{
        u64 now = os_time_now_micros();
        return (f64)(now - timer->last_tick_time) / 1000000.0;
}

u64 os_timer_delta_micros(struct os_timer *timer)
{
        u64 now = os_time_now_micros();
        return now - timer->last_tick_time;
}
