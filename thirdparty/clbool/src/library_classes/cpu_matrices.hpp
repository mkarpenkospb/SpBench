#pragma once
#include <utility>
#include <cstdint>
#include <vector>

void _aligned_free(void *_Memory);
void *_aligned_malloc(size_t _Size, size_t _Alignment);
// https://gist.github.com/donny-dont/1471329
template <typename T, std::size_t Alignment>
class aligned_allocator
{
public:

    // The following will be the same for virtually all allocators.
    typedef T * pointer;
    typedef const T * const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;
    typedef std::size_t size_type;
    typedef ptrdiff_t difference_type;

    T * address(T& r) const
    {
        return &r;
    }

    const T * address(const T& s) const
    {
        return &s;
    }

    std::size_t max_size() const
    {
        // The following has been carefully written to be independent of
        // the definition of size_t and to avoid signed/unsigned warnings.
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }


    // The following must be the same for all allocators.
    template <typename U>
    struct rebind
    {
        typedef aligned_allocator<U, Alignment> other;
    } ;

    bool operator!=(const aligned_allocator& other) const
    {
        return !(*this == other);
    }

    void construct(T * const p, const T& t) const
    {
        void * const pv = static_cast<void *>(p);

        new (pv) T(t);
    }

    void destroy(T * const p) const
    {
        p->~T();
    }

    // Returns true if and only if storage allocated from *this
    // can be deallocated from other, and vice versa.
    // Always returns true for stateless allocators.
    bool operator==(const aligned_allocator& other) const
    {
        return true;
    }


    // Default constructor, copy constructor, rebinding constructor, and destructor.
    // Empty for stateless allocators.
    aligned_allocator() { }

    aligned_allocator(const aligned_allocator&) { }

    template <typename U> aligned_allocator(const aligned_allocator<U, Alignment>&) { }

    ~aligned_allocator() { }


    // The following will be different for each allocator.
    T * allocate(const std::size_t n) const
    {
        // The return value of allocate(0) is unspecified.
        // Mallocator returns NULL in order to avoid depending
        // on malloc(0)'s implementation-defined behavior
        // (the implementation can define malloc(0) to return NULL,
        // in which case the bad_alloc check below would fire).
        // All allocators can return NULL in this case.
        if (n == 0) {
            return NULL;
        }

        // All allocators should contain an integer overflow check.
        // The Standardization Committee recommends that std::length_error
        // be thrown in the case of integer overflow.
        if (n > max_size())
        {
            throw std::length_error("aligned_allocator<T>::allocate() - Integer overflow.");
        }

        // Mallocator wraps malloc().
        void * const pv = _aligned_malloc(n * sizeof(T), Alignment);

        // Allocators should throw std::bad_alloc in the case of memory allocation failure.
        if (pv == NULL)
        {
            throw std::bad_alloc();
        }

        return static_cast<T *>(pv);
    }

    void deallocate(T * const p, const std::size_t n) const
    {
        _aligned_free(p);
    }


    // The following will be the same for all allocators that ignore hints.
    template <typename U>
    T * allocate(const std::size_t n, const U * /* const hint */) const
    {
        return allocate(n);
    }


    // Allocators are not required to be assignable, so
    // all allocators should have a private unimplemented
    // assignment operator. Note that this will trigger the
    // off-by-default (enabled under /Wall) warning C4626
    // "assignment operator could not be generated because a
    // base class assignment operator is inaccessible" within
    // the STL headers, but that warning is useless.
private:
    aligned_allocator& operator=(const aligned_allocator&);
};


using coordinates = std::pair<uint32_t, uint32_t>;
using matrix_coo_cpu_pairs = std::vector<coordinates>;
using cpu_buffer = std::vector<uint32_t, aligned_allocator<uint32_t, 64>>;
using cpu_buffer_f = std::vector<float, aligned_allocator<float, 64>>;

class matrix_dcsr_cpu {
    cpu_buffer _rows_pointers;
    cpu_buffer _rows_compressed;
    cpu_buffer _cols_indices;

public:
    matrix_dcsr_cpu(cpu_buffer rows_pointers, cpu_buffer rows_compressed, cpu_buffer cols_indices)
            : _rows_pointers(std::move(rows_pointers)), _rows_compressed(std::move(rows_compressed)),
              _cols_indices(std::move(cols_indices)) {}

    matrix_dcsr_cpu() = default;

    matrix_dcsr_cpu &operator=(matrix_dcsr_cpu other) {
        _rows_pointers = std::move(other._rows_pointers);
        _rows_compressed = std::move(other._rows_compressed);
        _cols_indices = std::move(other._cols_indices);
        return *this;
    }

    cpu_buffer &rows_pointers() {
        return _rows_pointers;
    }

    cpu_buffer &rows_compressed() {
        return _rows_compressed;
    }

    cpu_buffer &cols_indices() {
        return _cols_indices;
    }

    const cpu_buffer &rows_pointers() const {
        return _rows_pointers;
    }

    const cpu_buffer &rows_compressed() const {
        return _rows_compressed;
    }

    const cpu_buffer &cols_indices() const {
        return _cols_indices;
    }

};


class matrix_coo_cpu {
    cpu_buffer _rows_indices;
    cpu_buffer _cols_indices;

public:
    matrix_coo_cpu(cpu_buffer rows_indices, cpu_buffer cols_indices)
            : _rows_indices(std::move(rows_indices))
            , _cols_indices(std::move(cols_indices))
            {}

    matrix_coo_cpu() = default;

    matrix_coo_cpu &operator=(matrix_coo_cpu other) {
        _rows_indices = std::move(other._rows_indices);
        _cols_indices = std::move(other._cols_indices);
        return *this;
    }

    cpu_buffer &rows_indices() {
        return _rows_indices;
    }

    cpu_buffer &cols_indices() {
        return _cols_indices;
    }

    const cpu_buffer &rows_indices() const {
        return _rows_indices;
    }

    const cpu_buffer &cols_indices() const {
        return _cols_indices;
    }

};