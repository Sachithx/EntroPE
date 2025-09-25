import fsspec
from luigi.target import FileSystem, FileSystemTarget


class FSSpecFileSystem(FileSystem):
    def __init__(self, fs: fsspec.AbstractFileSystem):
        self.fs = fs

    def exists(self, path):
        return self.fs.exists()

    def remove(self, path, recursive=True, skip_trash=True):
        raise NotImplementedError()

    def isdir(self, path):
        return self.fs.isdir(path)

    def listdir(self, path):
        return self.fs.ls(path)


class FSSpecTarget(FileSystemTarget):
    def __init__(self, path, fs: fsspec.AbstractFileSystem | None = None):
        self.path = path
        if fs is None:
            self.fsspec_fs = fsspec.filesystem("file")
        else:
            self.fsspec_fs = fs
        self._fs = None

    @property
    def fs(self):
        if self._fs is None:
            self._fs = FSSpecFileSystem(self.fsspec_fs)
        return self._fs

    def open(self, mode):
        return self.fs.open(self.path, mode=mode)
