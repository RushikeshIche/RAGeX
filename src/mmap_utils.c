#include "mmap_utils.h"

#ifdef _WIN32
#include <windows.h>

int mmap_file_read(const char* filepath, MmapFile* mf) {
    mf->hFile = CreateFileA(filepath, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (mf->hFile == INVALID_HANDLE_VALUE) return -1;
    
    LARGE_INTEGER size;
    if (!GetFileSizeEx(mf->hFile, &size)) {
        CloseHandle(mf->hFile);
        return -1;
    }
    mf->size = size.QuadPart;
    
    mf->hMapping = CreateFileMappingA(mf->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!mf->hMapping) { 
        CloseHandle(mf->hFile); 
        return -1; 
    }
    
    mf->data = MapViewOfFile(mf->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mf->data) { 
        CloseHandle(mf->hMapping); 
        CloseHandle(mf->hFile); 
        return -1; 
    }
    return 0;
}

void munmap_file(MmapFile* mf) {
    if (mf->data) UnmapViewOfFile(mf->data);
    if (mf->hMapping) CloseHandle(mf->hMapping);
    if (mf->hFile && mf->hFile != INVALID_HANDLE_VALUE) CloseHandle(mf->hFile);
}

#else
// Linux/Unix/MacOS/Termux
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int mmap_file_read(const char* filepath, MmapFile* mf) {
    mf->fd = open(filepath, O_RDONLY);
    if (mf->fd < 0) return -1;
    
    struct stat sb;
    if (fstat(mf->fd, &sb) < 0) {
        close(mf->fd);
        return -1;
    }
    mf->size = sb.st_size;
    
    mf->data = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->data == MAP_FAILED) { 
        close(mf->fd); 
        return -1; 
    }
    return 0;
}

void munmap_file(MmapFile* mf) {
    if (mf->data && mf->data != MAP_FAILED) munmap(mf->data, mf->size);
    if (mf->fd >= 0) close(mf->fd);
}
#endif
