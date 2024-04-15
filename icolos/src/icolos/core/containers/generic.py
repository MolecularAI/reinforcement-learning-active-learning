from shutil import copyfile
from distutils.dir_util import copy_tree
import json
import os
import sys
from typing import Iterable, Optional, Union, TypeVar
from copy import Error

from icolos.loggers.iologger import IOLogger
from icolos.utils.enums.logging_enums import LoggingConfigEnum

_LE = LoggingConfigEnum()
T = TypeVar("T")


class GenericData:
    """Container class to hold generic data of any file type"""

    def __init__(
        self,
        file_name: str,
        file_data: Optional[T]=None,
        argument: bool=True,
        file_id: Optional[int]=None,
        extension: Optional[str]=None,
    ):
        self._extension = (
            extension if extension is not None else file_name.split(".")[-1]
        )
        self._file_name = file_name
        self._file_data = file_data
        self._file_id = file_id
        # self._argument: bool = argument
        self._file_size = self.calculate_file_size()
        self._logger = IOLogger()

    def get_file_name(self) -> str:
        return self._file_name

    def get_data(self) -> T:
        return self._file_data

    def calculate_file_size(self) -> int:
        return sys.getsizeof(self._file_data)

    def get_extension(self) -> str:
        return self._extension

    def set_data(self, data: T):
        self._file_data = data

    def set_file_name(self, file_name):
        self._file_name = file_name

    def set_id(self, file_id):
        self._file_id = file_id

    def get_id(self) -> int:
        return self._file_id

    def set_extension(self, extension: str):
        self._extension = extension

    def write(self, path: str, join: bool=True, final_writeout: bool=False):
        """
        Handles all I/O operations for generic data.  Support for handling directories and symlinks
        """
        orig_path = path
        if join:
            path = os.path.join(path, self.get_file_name())
            self._logger.log(f"Using path {path}", _LE.DEBUG)

        if str(self._file_data).startswith("/"):
            self._logger.log(f"Handling file at {self._file_data}", _LE.DEBUG)
            # file data is a path, copy the file to the destination
            # if it's a file, its stored like this because it's large (> 2GB)
            if os.path.isfile(self._file_data):
                if not final_writeout:
                    self._logger.log(f"Symlinking file from {self._file_data} to {path}", _LE.DEBUG)
                    # if this is a writeout to a step, we can simply create a simlink
                    os.symlink(self._file_data, path, target_is_directory=False)
                else:
                    self._logger.log(f"Copying file from {self._file_data} to {path}", _LE.DEBUG)
                    # we cannot do this for the final writeout since /scratch or /tmp will eventually get cleaned
                    copyfile(self._file_data, path)

            elif os.path.isdir(self._file_data):
                self._logger.log(f"Copying directory from {self._file_data} to {orig_path}", _LE.DEBUG)
                # copy the entire directory to the parent dir
                copy_tree(self._file_data, orig_path)
        elif isinstance(self._file_data, list):
            self._logger.log(f"Writing list {path}", _LE.DEBUG)
            with open(path, "w") as f:
                f.writelines(self._file_data)

        elif isinstance(self._file_data, str):
            self._logger.log(f"Writing string {path}", _LE.DEBUG)
            with open(path, "w") as f:
                f.write(self._file_data)
        elif isinstance(self._file_data, dict):
            self._logger.log(f"Writing dict {path}", _LE.DEBUG)
            with open(path, "w") as f:
                f.write(json.dumps(self._file_data))
        else:
            self._logger.log(f"Writing other {path}", _LE.DEBUG)
            with open(path, "wb") as f:
                f.write(self._file_data)

    def update_data(self, data: T):
        if sys.getsizeof(data) != self._file_size:
            self.set_data(data)

    def __repr__(self) -> str:
        return f"GenericData object - name: {self._file_name}, extension: {self._extension}."

    def __str__(self) -> str:
        return self.__repr__()


class GenericContainer:
    """Container class to hold the instances of the Generic class, separated by extension"""

    def __init__(self):
        self._file_dict: dict[str, list[GenericData]] = {}

    # self._paths = []
    # self._strings = []

    def add_file(self, file: GenericData):
        ext = file.get_extension()
        file.set_id(self.get_next_file_id(ext))
        try:
            self._file_dict[ext].append(file)
        except NameError:
            self._initialise_list(ext)
            self._file_dict[ext].append(file)

    def _initialise_list(self, ext: str):
        self._file_dict[ext] = []

    def get_next_file_id(self, ext: str) -> int:
        ids = [file.get_id() for file in self.get_files_by_extension(ext)]
        if len(ids) == 0:
            return 0
        return max(ids) + 1

    def get_file_by_index(self, index: int) -> GenericData:
        for file in self.get_flattened_files():
            if file.get_id() == index:
                return file

    def add_files(self, files: list[GenericData]):
        extensions = list(set([f.get_extension() for f in files]))
        if len(extensions) > 1:
            raise Error("Cannot have more than one type of file")
        else:
            if extensions[0] in self._file_dict.keys():
                self._file_dict[extensions[0]].extend(files)
            else:
                self._file_dict[extensions[0]] = files

    def get_all_files(self) -> dict[str, list[GenericData]]:
        return self._file_dict

    def get_files_by_extension(self, ext: str) -> list[GenericData]:
        if ext in self._file_dict.keys():
            return self._file_dict[ext]
        self._initialise_list(ext)
        return self._file_dict[ext]

    def get_file_names_by_extension(self, ext: str) -> list[str]:
        try:
            return [f.get_file_name() for f in self._file_dict[ext]]
        except KeyError:
            self._initialise_list(ext)
            return [f.get_file_name() for f in self._file_dict[ext]]

    def get_file_types(self) -> Iterable[str]:
        return self._file_dict.keys()

    def get_flattened_files(self) -> list[GenericData]:
        rtn_files = []
        for key in self._file_dict.keys():
            for file in self._file_dict[key]:
                rtn_files.append(file)
        return rtn_files

    def get_file_by_name(self, name: str) -> GenericData:
        for file in self.get_flattened_files():
            if file.get_file_name() == name:
                return file

    def clear_file_dict(self):
        self._file_dict = {}

    # TODO this should be two separate functions depending on return type
    def get_argument_by_extension(
        self, ext: str, rtn_file_object: bool=False
    ) -> Union[GenericData, str]:
        files = []
        for file in self.get_flattened_files():
            if file.get_extension() == ext:
                files.append(file)
        assert len(files) > 0, f"No files with extension {ext} were found!"
        try:
            assert len(files) == 1
        except AssertionError:
            print(
                f"Found multiple files with extension {ext}, select the index of the file to be passed as an argument\n"
            )
            print("######################")
            for idx, file in enumerate(files):
                print(f"{idx}: {file.get_file_name()}")
            print("######################")
            index = input(">>> ")
            files = [files[int(index)]]

        if not rtn_file_object:
            return files[0].get_file_name()
        else:
            return files[0]

    def write_out_all_files(self, folder: str):
        """flattens all files in the container and writes to the specified directory"""
        for file in self.get_flattened_files():
            file.write(folder)
