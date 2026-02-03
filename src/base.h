#ifndef BASE_H_
#define BASE_H_

//////////////////////////////
// Feature detection macros

#if !defined(HAVE_ANDROID)
#if defined(__ANDROID__)
#define HAVE_ANDROID 1
#else
#define HAVE_ANDROID 0
#endif
#endif

#if !defined(HAVE_ARM)
#if defined(__arm__)
#define HAVE_ARM_32 1
#else
#define HAVE_ARM_32 0
#endif
#endif

#if !defined(HAVE_ARM_64)
#if defined(__aarch64__)
#define HAVE_ARM_64 1
#else
#define HAVE_ARM_64 0
#endif
#endif

#if !defined(HAVE_X86_32)
#if defined(__i386__) || (defined(_WIN32) && !defined(_WIN64))
#define HAVE_X86_32 1
#else
#define HAVE_X86_32 0
#endif
#endif

#if !defined(HAVE_X86_64)
#if defined(__x86_64__) || defined(_WIN64) || defined(_M_AMD64) || \
        defined(_M_X64) || defined(__amd64)
#define HAVE_X86_64 1
#else
#define HAVE_X86_64 0
#endif
#endif

#if !defined(HAVE_WINDOWS)
#if defined(_WIN32)
#define HAVE_WINDOWS 1
#else
#define HAVE_WINDOWS 0
#endif
#endif

#if !defined(_WIN32) && (defined(__unix__) || defined(__unix) || \
                         (defined(__APPLE__) && defined(__MACH__)))
#define HAVE_UNIX 1
#else
#define HAVE_UNIX 0
#endif

#if !defined(HAVE_LINUX)
#if defined(__linux__)
#define HAVE_LINUX 1
#else
#define HAVE_LINUX 0
#endif
#endif

#if !defined(HAVE_DEBUG)
#if defined(DEBUG) || defined(_DEBUG) || !defined(NDEBUG)
#define HAVE_DEBUG 1
#else
#define HAVE_DEBUG 0
#endif
#endif

#if !defined(HAVE_C99)
#if __STDC_VERSION__ >= 199901L
#define HAVE_C99 1
#else
#define HAVE_C99 0
#endif
#endif

#if !defined(HAVE_C11)
#if __STDC_VERSION__ >= 201112L
#define HAVE_C11 1
#else
#define HAVE_C11 0
#endif
#endif

#if !defined(HAVE_CPP)
#if defined(__cplusplus)
#define HAVE_CPP 1
#else
#define HAVE_CPP 0
#endif
#endif

#if !defined(HAVE_MSVC)
#if defined(_MSC_VER) && !defined(__clang__)
#define HAVE_MSVC 1
#else
#define HAVE_MSVC 0
#endif
#endif

#if !defined(HAVE_CLANG)
#if defined(__clang__)
#define HAVE_CLANG 1
#else
#define HAVE_CLANG 0
#endif
#endif

#if !defined(HAVE_GCC)
#if defined(__GNUC__) && !defined(__clang__)
#define HAVE_GCC 1
#else
#define HAVE_GCC 0
#endif
#endif

#if !defined(HAVE_TCC)
#if defined(__TINYC__)
#define HAVE_TCC 1
#else
#define HAVE_TCC 0
#endif
#endif

#if !defined(HAVE_ASAN)
#if HAVE_CLANG
#if defined(__has_feature)
#if __has_feature(address_sanitizer) || defined(__SANITIZE_ADDRESS__)
#define HAVE_ASAN 1
#endif
#endif
#elif HAVE_MSVC
#if defined(__SANITIZE_ADDRESS__)
#define HAVE_ASAN 1
#endif
#endif
#endif

#if !defined(HAVE_ASAN)
// If HAVE_ASAN is not defined at this point, we assume it is not available.
#define HAVE_ASAN 0
#endif

#if !defined(HAVE_INSTRSET)
#if defined(__AVX512VL__) && defined(__AVX512BW__) && defined(__AVX512DQ__)
#define HAVE_INSTRSET 10
#elif defined(__AVX512F__) || defined(__AVX512__)
#define HAVE_INSTRSET 9
#elif defined(__AVX2__)
#define HAVE_INSTRSET 8
#elif defined(__AVX__)
#define HAVE_INSTRSET 7
#elif defined(__SSE4_2__)
#define HAVE_INSTRSET 6
#elif defined(__SSE4_1__)
#define HAVE_INSTRSET 5
#elif defined(__SSSE3__)
#define HAVE_INSTRSET 4
#elif defined(__SSE3__)
#define HAVE_INSTRSET 3
#elif defined(__SSE2__) || HAVE_X86_64 == 1
#define HAVE_INSTRSET 2
#elif defined(__SSE__)
#define HAVE_INSTRSET 1
#else
#define HAVE_INSTRSET 0
#endif
#endif

#ifndef HAVE_STB_SPRINTF
#if __has_include(<stb_sprintf.h>)
#define HAVE_STB_SPRINTF 1
#else
#define HAVE_STB_SPRINTF 0
#endif
#endif

#ifndef HAVE_STB_IMAGE
#if __has_include(<stb_image.h>)
#define HAVE_STB_IMAGE 1
#else
#define HAVE_STB_IMAGE 0
#endif
#endif

#ifndef HAVE_VULKAN
#if __has_include(<vulkan/vulkan.h>)
#define HAVE_VULKAN 1
#else
#define HAVE_VULKAN 0
#endif
#endif

#if HAVE_GCC || HAVE_CLANG
typedef __builtin_va_list va_list;
#define va_start(ap, last) __builtin_va_start(ap, last)
#define va_end(ap) __builtin_va_end(ap)
#define va_arg(ap, type) __builtin_va_arg(ap, type)
#define va_copy(dest, src) __builtin_va_copy(dest, src)
#else
#include <stdarg.h>
#endif

//////////////////////////////
// Type definitions

#define F32_MAX 3.402823466e+38f
#define F32_MIN 1.175494351e-38f

#define F64_MAX 1.7976931348623158e+308
#define F64_MIN 2.2250738585072014e-308

#define I8_MAX 127
#define I8_MIN (-128)

#define I16_MAX 32767
#define I16_MIN (-32768)

#define I32_MAX 2147483647
#define I32_MIN (-2147483648)

#define I64_MAX 9223372036854775807ll
#define I64_MIN (-9223372036854775808ll)

#define U8_MAX 255
#define U8_MIN 0

#define U16_MAX 65535
#define U16_MIN 0

#define U32_MAX 4294967295u
#define U32_MIN 0

#define U64_MAX 18446744073709551615ull
#define U64_MIN 0

#if HAVE_ARM_64 || HAVE_X86_64
#define USIZE_MAX U64_MAX
#define USIZE_MIN U64_MIN
#define ISIZE_MAX I64_MAX
#define ISIZE_MIN I64_MIN
#elif HAVE_ARM_32 || HAVE_X86_32
#define USIZE_MAX U32_MAX
#define USIZE_MIN U32_MIN
#define ISIZE_MAX I32_MAX
#define ISIZE_MIN I32_MIN
#endif

#define F32_EPSILON 1.19209e-07f
#define F64_EPSILON 2.22045e-16

#define F32_PI 3.14159265358979323846f
#define F64_PI 3.14159265358979323846

#if HAVE_CPP || __STDC_VERSION__ >= 199901L
#include <stdbool.h>
typedef bool b8;
#else
#define true 1
#define false 0
typedef unsigned char b8;
#endif

typedef int b32;

typedef float f32;
typedef double f64;

typedef signed char i8;
typedef signed short i16;
typedef signed int i32;
#if HAVE_WINDOWS
typedef signed long long i64;
#elif HAVE_ANDROID
typedef signed long long i64;
#else
typedef signed long i64;
#endif

typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;

#if HAVE_WINDOWS || HAVE_ANDROID
typedef unsigned long long u64;
#else
typedef unsigned long u64;
#endif

#if HAVE_ARM_64
typedef unsigned long usize;
#elif HAVE_ARM_32
typedef unsigned int usize;
#elif HAVE_X86_64
#if HAVE_ANDROID
typedef unsigned long usize;
#else
typedef u64 usize;
#endif
#elif HAVE_X86_32
typedef u32 usize;
#else
#error "Unsupported architecture"
#endif

#if HAVE_ARM_64
typedef signed long isize;
#elif HAVE_ARM_32
typedef signed int isize;
#elif HAVE_X86_64
typedef i64 isize;
#elif HAVE_X86_32
typedef i32 ISize;
#else
#error "Unsupported architecture"
#endif

#if HAVE_X86_64 || HAVE_ARM_64
typedef u64 uptr;
typedef i64 iptr;
#elif HAVE_X86_32 || HAVE_ARM_32
typedef u32 uptr;
typedef i32 iptr;
#else
#error "Unsupported architecture"
#endif

// Properly define static assertions.
#define STATIC_ASSERT(cond, msg) \
        typedef char static_assertion__##msg[(cond) ? 1 : -1]

STATIC_ASSERT(sizeof(u8) == 1, expected_u8_to_be_1_byte);
STATIC_ASSERT(sizeof(u16) == 2, expected_u16_to_be_2_bytes);
STATIC_ASSERT(sizeof(u32) == 4, expected_u32_to_be_4_bytes);
STATIC_ASSERT(sizeof(u64) == 8, expected_u64_to_be_8_bytes);
STATIC_ASSERT(sizeof(i8) == 1, expected_i8_to_be_1_byte);
STATIC_ASSERT(sizeof(i16) == 2, expected_I16_to_be_2_bytes);
STATIC_ASSERT(sizeof(i32) == 4, expected_I32_to_be_4_bytes);
STATIC_ASSERT(sizeof(i64) == 8, expected_I64_to_be_8_bytes);
STATIC_ASSERT(sizeof(f32) == 4, expected_F32_to_be_4_bytes);
STATIC_ASSERT(sizeof(f64) == 8, expected_F64_to_be_8_bytes);

//////////////////////////////
// Printf format macros

#if HAVE_WINDOWS || HAVE_ANDROID
#define PRIuz "llu"
#define PRIiz "lli"
#define PRIu64 "llu"
#define PRIi64 "lli"
#else
#define PRIuz "lu"
#define PRIiz "li"
#define PRIu64 "lu"
#define PRIi64 "li"
#endif

//////////////////////////////
// Safe casting macros

#if HAVE_DEBUG
#define F64_TO_F32(x)                                                \
        ((x) >= (f64) - F32_MAX && (x) <= (f64)F32_MAX ?             \
                 (f32)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,       \
                            "Value " #x " is out of range for f32"), \
                  0.0f))

#define I32_TO_U32(x)                                                      \
        ((x) >= 0 ? (u32)(x) :                                             \
                    (fail_func(__FILE_NAME__, __LINE__, __func__,          \
                               "Value " #x " is negative, cannot cast to " \
                               "u32"),                                     \
                     0))

#define I32_TO_USIZE(x)                                                    \
        ((x) >= 0 ? (usize)(x) :                                           \
                    (fail_func(__FILE_NAME__, __LINE__, __func__,          \
                               "Value " #x " is negative, cannot cast to " \
                               "usize"),                                   \
                     0))

#define U32_TO_I32(x)                                                \
        ((x) <= (u32)I32_MAX ?                                       \
                 (i32)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,       \
                            "Value " #x " is out of range for i32"), \
                  0))

#define F64_TO_USIZE(x)                                                \
        ((x) >= 0.0 && (x) <= (f64)USIZE_MAX ?                         \
                 (usize)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,         \
                            "Value " #x " is out of range for usize"), \
                  0))

#define USIZE_TO_U32(x)                                              \
        ((x) <= (usize)U32_MAX ?                                     \
                 (u32)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,       \
                            "Value " #x " is out of range for u32"), \
                  0))

#define USIZE_TO_I32(x)                                              \
        ((x) <= (usize)I32_MAX ?                                     \
                 (i32)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,       \
                            "Value " #x " is out of range for i32"), \
                  0))

#define USIZE_TO_I64(x)                                              \
        ((x) <= (usize)I64_MAX ?                                     \
                 (i64)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,       \
                            "Value " #x " is out of range for i64"), \
                  0))

#define USIZE_TO_F32(x)                                              \
        ((x) <= (usize)F32_MAX ?                                     \
                 (f32)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,       \
                            "Value " #x " is out of range for f32"), \
                  0.0f))

#define F64_EXACT_UINT_MAX ((usize)1 << 53)
#define USIZE_TO_F64(x)                                               \
        ((x) <= F64_EXACT_UINT_MAX ?                                  \
                 (f64)(x) :                                           \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,        \
                            "Value " #x                               \
                            " cannot be represented exactly as f64"), \
                  0.0))

#define USIZE_TO_ISIZE(x)                                              \
        ((x) <= (usize)ISIZE_MAX ?                                     \
                 (isize)(x) :                                          \
                 (fail_func(__FILE_NAME__, __LINE__, __func__,         \
                            "Value " #x " is out of range for isize"), \
                  0))

#define ISIZE_TO_USIZE(x)                                          \
        ((x) >= 0 ? (usize)(x) :                                   \
                    (fail_func(__FILE_NAME__, __LINE__, __func__,  \
                               "Value " #x " is negative, cannot " \
                               "cast to usize"),                   \
                     0))

#else
#define F64_TO_F32(x) ((f32)(x))
#define I32_TO_USIZE(x) ((usize)(x))
#define U32_TO_I32(x) ((i32)(x))
#define F64_TO_USIZE(x) ((usize)(x))
#define USIZE_TO_U32(x) ((u32)(x))
#define USIZE_TO_I32(x) ((i32)(x))
#define USIZE_TO_I64(x) ((i64)(x))
#define USIZE_TO_F32(x) ((f32)(x))
#define USIZE_TO_F64(x) ((f64)(x))
#define USIZE_TO_ISIZE(x) ((isize)(x))
#define ISIZE_TO_USIZE(x) ((usize)(x))
#endif

//////////////////////////////
// General macros

#ifndef NULL
#define NULL 0
#endif

#if !HAVE_WINDOWS
#define MAX_PATH 256;
#endif
#define MAX_NAME 64;

#if HAVE_MSVC
#define align_of(type) __alignof(type)
#elif HAVE_CLANG
#define align_of(type) __alignof(type)
#elif HAVE_GCC || HAVE_TCC
#define align_of(type) __alignof__(type)
#else
#error "Unsupported compiler"
#endif

#if HAVE_MSVC
#define THREAD_LOCAL __declspec(thread)
#elif HAVE_GCC || HAVE_CLANG
#define THREAD_LOCAL __thread
#else
#error "Unsupported compiler"
#endif

#if HAVE_CPP
#define C_LINKAGE_BEGIN extern "C" {
#define C_LINKAGE_END }
#define C_LINKAGE extern "C"
#else
#define C_LINKAGE_BEGIN
#define C_LINKAGE_END
#define C_LINKAGE
#endif

#if HAVE_WINDOWS
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API
#endif

#if HAVE_GCC || HAVE_CLANG
#define PRINTF_LIKE(fmt_idx, first_arg) \
        __attribute__((format(printf, fmt_idx, first_arg)))
#else
#define PRINTF_LIKE(fmt_idx, first_arg)
#endif

#if HAVE_ASAN
C_LINKAGE void __asan_poison_memory_region(const volatile void *addr,
                                           size_t size);
C_LINKAGE void __asan_unpoison_memory_region(const volatile void *addr,
                                             size_t size);
#define asan_poison_memory_region(addr, size) \
        __asan_poison_memory_region((addr), (size))
#define asan_unpoison_memory_region(addr, size) \
        __asan_unpoison_memory_region((addr), (size))
#else
#define asan_poison_memory_region(addr, size) ((void)(addr), (void)(size))
#define asan_unpoison_memory_region(addr, size) ((void)(addr), (void)(size))
#endif

//////////////////////////////
// Convinience macros

#define kb(n) (((u64)(n)) << 10)
#define mb(n) (((u64)(n)) << 20)
#define gb(n) (((u64)(n)) << 30)
#define tb(n) (((u64)(n)) << 40)
#define thousand(n) ((n) * 1000)
#define million(n) ((n) * 1000000)
#define billion(n) ((n) * 1000000000)

#define min(a, b) ((a) < (b) ? (a) : (b))
#define max(a, b) ((a) > (b) ? (a) : (b))
#define clamp(a, min, max) min(max(a, min), max)
#define clamp01(a) min(max(a, 0), 1)
#define clamp_top(a, max) min(a, max)
#define clamp_bottom(a, min) max(a, min)
#define lerp(a, b, t) ((a) + ((b) - (a)) * (t))
#define saturate(a) clamp01(a)

#define align_pow2(x, b) (((x) + (b) - 1) & (~((b) - 1)))
#define align_down_pow2(x, b) ((x) & (~((b) - 1)))
#define align_pad_pow2(x, b) ((0 - (x)) & ((b) - 1))
#define is_pow2(x) ((x) != 0 && ((x) & ((x) - 1)) == 0)
#define is_pow2_or_zero(x) ((((x) - 1) & (x)) == 0)

#define ARRAY_COUNT(array) (i32)((sizeof(array) / sizeof(array[0])))

#define memory_copy_struct(d, s) \
        memory_copy((d), sizeof(*(d)), (s), sizeof(*(d)))
#define memory_copy_array(d, s) memory_copy((d), sizeof(d), (s), sizeof(d))
#define memory_copy_typed(d, s, c) \
        memory_copy((d), sizeof(d), (s), sizeof(*(d)) * (c))

//////////////////////////////
// General

struct base_state;

b8 base_init(const char *org_name, const char *app_name, usize memory_size_mb);
b8 base_init_from_external(struct base_state *external_state);
void base_shutdown(void);
void base_shutdown_from_external(void);

struct base_state *get_base_state(void);

//////////////////////////////
// Memory

void *memory_alloc(usize size);
void *memory_realloc(void *ptr, usize old_size, usize new_size);
void memory_release(void *ptr);

void *memory_alloc_aligned(usize size, usize alignment);
void memory_release_aligned(void *ptr);

void *memory_set(void *ptr, usize ptr_size, u8 value, usize size);
void *memory_zero(void *ptr, usize size);
void *memory_copy(void *dest, usize dest_size, const void *src, usize src_size);
b8 memory_equal(const void *ptr1, usize ptr1Size, void *ptr2, usize ptr2Size);

//////////////////////////////
// Formatting and printing

i32 format_cstr(char *buffer, usize buffer_size, const char *fmt, ...)
        PRINTF_LIKE(3, 4);
i32 format_cstr_va(char *buffer, usize buffer_size, const char *fmt,
                   va_list args);

void print_fmt(const char *fmt, ...) PRINTF_LIKE(1, 2);
void print_error_fmt(const char *fmt, ...) PRINTF_LIKE(1, 2);

//////////////////////////////
// Logging

enum log_level {
        LOG_LEVEL_DEBUG = 0,
        LOG_LEVEL_INFO,
        LOG_LEVEL_ERROR,
};

void log_set_level(enum log_level kind);
void log_set_colored_output(b8 enable);
void log_message(enum log_level level, const char *message, u64 length);
void log_message_fmt(enum log_level level, const char *fmt, ...)
        PRINTF_LIKE(2, 3);

#define LOG_DEBUG(...) log_message_fmt(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define LOG_INFO(...) log_message_fmt(LOG_LEVEL_INFO, __VA_ARGS__)
#define LOG_ERROR(...) log_message_fmt(LOG_LEVEL_ERROR, __VA_ARGS__)

//////////////////////////////
// Assertions

#if HAVE_MSVC
#include <intrin.h>
#define __FILE_NAME__ (cstr_find_last_char("\\" __FILE__, '\\') + 1)
#elif HAVE_TCC
#define __FILE_NAME__ __FILE__
#elif HAVE_GCC || HAVE_CLANG
// GCC and Clang provide __FILE_NAME__ natively.
#else
#error "Unsupported compiler"
#endif

#define UNUSED(x) ((void)(x))

#if HAVE_MSVC
#define TRAP()                  \
        do {                    \
                __debugbreak(); \
                __assume(0);    \
        } while (0)
#elif HAVE_CLANG || HAVE_GCC
#define TRAP() __builtin_trap()
#else
#define TRAP() (*(volatile int *)0 = 0)
#endif

// Ensure is available in all builds, even in release builds.
#define ENSURE(expr)                                                         \
        do {                                                                 \
                if (!(expr)) {                                               \
                        LOG_ERROR("Ensure failed in %s:%d %s(): %s",         \
                                  __FILE_NAME__, __LINE__, __func__, #expr); \
                        TRAP();                                              \
                }                                                            \
        } while (false)

#define ENSURE_MSG(expr, ...)                                                \
        do {                                                                 \
                if (!(expr)) {                                               \
                        LOG_ERROR("Ensure failed in %s:%d %s(): %s",         \
                                  __FILE_NAME__, __LINE__, __func__, #expr); \
                        LOG_ERROR(__VA_ARGS__);                              \
                        TRAP();                                              \
                }                                                            \
        } while (false)

#if HAVE_DEBUG
#define ASSERT(expr)                                                         \
        do {                                                                 \
                if (!(expr)) {                                               \
                        LOG_ERROR("Assertion failed in %s:%d %s(): %s",      \
                                  __FILE_NAME__, __LINE__, __func__, #expr); \
                        TRAP();                                              \
                }                                                            \
        } while (false)

#define ASSERT_MSG(expr, ...)                                                \
        do {                                                                 \
                if (!(expr)) {                                               \
                        LOG_ERROR("Assertion failed in %s:%d %s(): %s",      \
                                  __FILE_NAME__, __LINE__, __func__, #expr); \
                        LOG_ERROR(__VA_ARGS__);                              \
                        TRAP();                                              \
                }                                                            \
        } while (false)

#else
#define ASSERT(expr) ((void)0)
#define ASSERT_MSG(expr, ...) ((void)0)
#endif

#define FAIL()                                                              \
        do {                                                                \
                LOG_ERROR("Failure in %s:%d %s()", __FILE_NAME__, __LINE__, \
                          __func__);                                        \
                TRAP();                                                     \
        } while (false)

#define FAIL_MSG(...)                                                       \
        do {                                                                \
                LOG_ERROR("Failure in %s:%d %s()", __FILE_NAME__, __LINE__, \
                          __func__);                                        \
                                                                            \
                LOG_ERROR(__VA_ARGS__);                                     \
                TRAP();                                                     \
        } while (false)

#define UNREACHABLE()                                                \
        do {                                                         \
                LOG_ERROR("Unreachable point reached in %s:%d %s()", \
                          __FILE_NAME__, __LINE__, __func__);        \
                TRAP();                                              \
        } while (false)

#define UNIMPLEMENTED()                                                \
        do {                                                           \
                LOG_ERROR("Unimplemented point reached in %s:%d %s()", \
                          __FILE_NAME__, __LINE__, __func__);          \
                TRAP();                                                \
        } while (false)

void fail_func(const char *file, int line, const char *func,
               const char *message);

//////////////////////////////
// Arena

#define ARENA_HEADER_SIZE 128
#define ARENA_USE_FREE_LIST 1
#define ARENA_USE_MALLOC 0
#define ARENA_DEFAULT_RESERVE_SIZE mb(1)
#define ARENA_DEFAULT_COMMIT_SIZE kb(64)

enum arena_flag {
        ARENA_FLAG_NONE = 0,
        /// Wether or not the arena is allowed to allocate new blocks of memory.
        ARENA_FLAG_NO_CHAIN = 1 << 0,
};
typedef u32 arena_flags;

/// The arena allows to allocate dynamic memory efficiently.
/// A large chunk of memory can be reserved upfront but will only be committed if needed.
/// The arena will never reallocate memory that is currently in use.
/// If not enough memory is available, a new arena will be created and linked to the current one.
struct arena {
        /// Previous arena in the chain.
        struct arena *prev;
        /// Current arena in the chain. (Unsure, but maybe current should be named next?)
        /// If its the head of the chain it points to itself. If its somewhere in the chain it points to the next
        /// arena (block).
        struct arena *current;
        /// Arena flags.
        arena_flags flags;
        /// This one may be removed, because it seems not really needed/used.
        u64 committed_size;
        /// The size of the reserved memory of the whole arena.
        u64 reserved_size;
        /// The base position of the this arena (block) in the chain. This in combination with position gives the
        /// absolute position of the memory.
        u64 base_position;
        /// Position up to which memory has been allocated.
        u64 position;
        /// The position up to which memory has been committed.
        u64 commit;
        /// The position up to which memory has been reserved.
        u64 reserved;
        /// The size of the reserved memory in the deallocated blocks that can be used again.
        u64 free_size;
        /// The last block that was deallocated and is free to be used again.
        struct arena *free_last;
};
STATIC_ASSERT(sizeof(struct arena) <= ARENA_HEADER_SIZE,
              expected_arena_header_to_be_smaller_than_128_bytes);

struct arena_config {
        /// Memory in bytes that should be reserved for the arena.
        u64 reserve_size;
        /// Memory in bytes that should be committed for the arena by default.
        u64 commit_size;
        /// Configuration flags.
        arena_flags flags;
};

struct arena_temp {
        /// The arena to which the temporary scope belongs.
        struct arena *arena;
        /// The position where the temporary scope started.
        u64 position;
};

/// Allocate a new arena with the given configuration.
struct arena *arena_alloc_from_config(struct arena_config *config);

/// Allocate a new arena with default configuration.
struct arena *arena_alloc(void);

/// Release the arena and all its blocks.
void arena_release(struct arena *arena);

/// Allocate size bytes of memory in the arena with the given alignment.
void *arena_push(struct arena *arena, u64 size, u64 align);

/// Allocate a copy of the given C string in the arena.
char *arena_push_cstr(struct arena *arena, const char *cstr);

/// Allocate a formatted C string in the arena.
char *arena_push_cstr_fmt(struct arena *arena, const char *fmt, ...);

/// Allocate a formatted C string in the arena with va_list.
char *arena_push_cstr_fmt_va(struct arena *arena, const char *fmt,
                             va_list args);

/// Get the current position in the arena.
u64 arena_position(struct arena *arena);

/// Pop/Free all the memory in the arena up to the given position.
void arena_pop_to(struct arena *arena, u64 position);

/// Clear all the memory in the arena.
void arena_clear(struct arena *arena);

/// Free/Pop size bytes from the arena.
void arena_pop(struct arena *arena, u64 size);

/// Get the base address of the arena. This is the address where the users memory starts.
void *arena_get_base(struct arena *arena);

/// Starts a temporary scope in the arena. Passing the returned arena_temp to arena_temp_end will pop all the
/// memory that was allocated between the two calls.
struct arena_temp arena_temp_begin(struct arena *arena);

/// Ends a temporary scope in the arena, freeing all memory that was allocated since the corresponding
/// call to arena_temp_begin.
void arena_temp_end(struct arena_temp temp);

/// Starts a scratch arena for temporary allocations.
struct arena_temp arena_scratch_begin(struct arena **conflicts,
                                      i32 conflictCount);

/// Ends a scratch arena, freeing all memory that was allocated in the scratch arena.
void arena_scratch_end(struct arena_temp scratch);

#define arena_push_array_aligned_no_zero(arena, T, count, align) \
        (T *)arena_push((arena), sizeof(T) * (count), (align))

#define arena_push_array_aligned(arena, T, count, align)                       \
        (T *)memory_zero(arena_push_array_aligned_no_zero((arena), T, (count), \
                                                          (align)),            \
                         sizeof(T) * (count))

#define arena_push_array_no_zero(arena, T, count)             \
        arena_push_array_aligned_no_zero((arena), T, (count), \
                                         max(8, align_of(T)))

#define arena_push_array(arena, T, count) \
        arena_push_array_aligned((arena), T, (count), max(8, align_of(T)))

#define arena_push_struct_no_zero(arena, T) \
        arena_push_array_aligned_no_zero((arena), T, 1, max(8, align_of(T)))

#define arena_push_struct(arena, T) \
        arena_push_array_aligned((arena), T, 1, max(8, align_of(T)))

//////////////////////////////
// C String

usize cstr_length(const char *str);
b8 cstr_is_digit(char c);
b8 cstr_is_alpha(char c);
b8 cstr_equal(const char *a, const char *b);
b8 cstr_equaln(const char *a, const char *b, u64 n);
void cstr_copy(char *dest, const char *src, u64 destSize);
u64 cstr_hash(const char *str);
char *cstr_clone(const char *str);
char *cstr_ends_with(const char *str, const char *suffix);
char *cstr_starts_with(const char *str, const char *prefix);
char *cstr_find_last_char(const char *str, char c);
char *cstr_find_char(const char *str, char c);

i32 cstr_to_i32(const char *str);
f32 cstr_to_f32(const char *str);

//////////////////////////////
// String View

struct string_view {
        const char *str;
        usize length;
};

struct string_view string_view_from_parts(const char *data, usize length);
struct string_view string_view_from_cstr(const char *cstr);

b8 string_view_equal(struct string_view a, struct string_view b);
b8 string_view_equal_cstr(struct string_view view, const char *cstr);

//////////////////////////////
// Lexer

enum token_type {
        TOKEN_TYPE_UNKNOWN = 0,
        TOKEN_TYPE_OPEN_PAREN,
        TOKEN_TYPE_CLOSE_PAREN,
        TOKEN_TYPE_OPEN_BRACKET,
        TOKEN_TYPE_CLOSE_BRACKET,
        TOKEN_TYPE_OPEN_BRACE,
        TOKEN_TYPE_CLOSE_BRACE,
        TOKEN_TYPE_OPEN_ANGLE_BRACKET,
        TOKEN_TYPE_CLOSE_ANGLE_BRACKET,
        TOKEN_TYPE_COLON,
        TOKEN_TYPE_SEMICOLON,
        TOKEN_TYPE_ASTERISK,
        TOKEN_TYPE_EQUAL,
        TOKEN_TYPE_PLUS,
        TOKEN_TYPE_MINUS,
        TOKEN_TYPE_DIVIDE,
        TOKEN_TYPE_HASH,
        TOKEN_TYPE_COMMA,
        TOKEN_TYPE_STRING,
        TOKEN_TYPE_IDENTIFIER,
        TOKEN_TYPE_NUMBER,
        TOKEN_TYPE_END_OF_STREAM,
};

enum comment_type {
        COMMENT_TYPE_DOUBLE_SLASH = 0,
        COMMENT_TYPE_HASH,
};

struct token {
        enum token_type type;

        usize line;
        usize column;

        struct string_view text;
        f64 number;
};

struct lexer {
        const char *filePath;
        const char *source;
        usize source_length;
        char *position;
        usize line;
        usize column;
        enum comment_type comment_type;
        struct token last_token;
        b8 is_revert_token;
        b8 is_eof;
};

struct lexer lexer_init(const char *file_path, const char *source);

struct token lexer_next_token(struct lexer *lexer);
void lexer_revert_token(struct lexer *lexer);
struct token lexer_peek_token(struct lexer *lexer);

b8 lexer_expect_identifier(struct lexer *lexer, const char *identifier,
                           struct token *out_token, b8 fatal);
b8 lexer_expect_token_kind(struct lexer *lexer, enum token_type type,
                           struct token *out_token, b8 fatal);
b8 lexer_expect_string(struct lexer *lexer, const char *str,
                       struct token *out_token, b8 fatal);
b8 lexer_expect_number(struct lexer *lexer, f64 number, struct token *out_token,
                       b8 fatal);

//////////////////////////////
// SD file format parser

enum sd_node_type {
        SD_NODE_TYPE_UNKNOWN = 0,
        SD_NODE_TYPE_OBJECT,
        SD_NODE_TYPE_ARRAY,
        SD_NODE_TYPE_STRING,
        SD_NODE_TYPE_NUMBER,
        SD_NODE_TYPE_BOOLEAN,
};

struct sd_node {
        enum sd_node_type type;
        // Empty string if unnamed
        struct string_view name;

        // Sibling pointer
        struct sd_node *next;
        // For objects/arrays
        struct sd_node *first_child;
        struct sd_node *last_child;

        // Cached count for arrays
        u32 child_count;

        union {
                struct string_view str;
                f64 number;
                b8 boolean;
        } value;
};

struct sd_parser *sd_parse(struct arena *arena, const char *file_path,
                           const char *source);

b8 sd_parser_has_errors(struct sd_parser *parser);

struct sd_node *sd_parser_root(struct sd_parser *parser);

struct sd_node *sd_node_find(struct sd_node *node, const char *name);
b8 sd_node_find_object(struct sd_node *node, const char *name,
                       struct sd_node **outObject);
b8 sd_node_find_array(struct sd_node *node, const char *name,
                      struct sd_node **outArray);
b8 sd_node_find_string(struct sd_node *node, const char *name,
                       const char *defaultValue, struct string_view *outValue);
b8 sd_node_find_number(struct sd_node *node, const char *name, f64 defaultValue,
                       f64 *outValue);
b8 sd_node_find_boolean(struct sd_node *node, const char *name, b8 defaultValue,
                        b8 *outValue);

b8 sd_node_is_object(struct sd_node *node);
b8 sd_node_is_array(struct sd_node *node);
b8 sd_node_is_string(struct sd_node *node);
b8 sd_node_is_number(struct sd_node *node);
b8 sd_node_is_boolean(struct sd_node *node);

b8 sd_node_object(struct sd_node *node, struct sd_node **outObject);
b8 sd_node_array(struct sd_node *node, struct sd_node **outArray);
b8 sd_node_string(struct sd_node *node, const char *defaultValue,
                  struct string_view *outValue);
b8 sd_node_number(struct sd_node *node, f64 defaultValue, f64 *outValue);
b8 sd_node_boolean(struct sd_node *node, b8 defaultValue, b8 *outValue);

//////////////////////////////
// OS

struct os_system_info {
        u32 logical_processor_count;
        u64 page_size;
        u64 large_page_size;
        u64 allocation_granularity;
};

void os_get_system_info(struct os_system_info **outInfo);

void *os_memory_reserve(usize size);
void os_memory_release(void *ptr, usize size);
void os_memory_commit(void *ptr, usize size);
void os_memory_decommit(void *ptr, usize size);

//////////////////////////////
// File

typedef u64 os_file_handle;
typedef u32 os_access_flags;

enum os_access_flag {
        OS_ACCESS_FLAG_READ = 1 << 0,
        OS_ACCESS_FLAG_WRITE = 1 << 1,
        OS_ACCESS_FLAG_APPEND = 1 << 2,
};

struct os_file_stats {
        u64 size;
        u64 last_modified_time;
};

b8 os_file_read_all_cstr(const char *path, struct arena *arena, void **out_data,
                         u64 *out_size);
b8 os_file_read_all_string_cstr(const char *path, struct arena *arena,
                                char **out_string);

//////////////////////////////
// Time

struct os_timer {
        u64 start_time;
        u64 last_tick_time;
};

u64 os_time_now_micros(void);
f64 os_time_now_seconds(void);

void os_timer_start(struct os_timer *timer);
f64 os_timer_elapsed_seconds(struct os_timer *timer);
u64 os_timer_elapsed_micros(struct os_timer *timer);

void os_timer_tick(struct os_timer *timer);
f64 os_timer_delta_seconds(struct os_timer *timer);
u64 os_timer_delta_micros(struct os_timer *timer);

#endif // BASE_H
