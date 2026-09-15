import logging
import os
import re
import tarfile
import sys
import unix_ar

logger = logging.getLogger(__name__)


class ControlFile(object):
    def __init__(self, control):
        self._raw_control_data = control

    @property
    def control_data(self):
        return self._raw_control_data

    @property
    def package(self):
        match = re.search(r'^Package: (.*)$', self._raw_control_data, re.M)
        return match.group(1)

    @property
    def source(self):
        match = re.search(r'^Source: (.*)$', self._raw_control_data, re.M)
        return match.group(1) if match else self.package

    @property
    def version(self):
        match = re.search(r'^Version: (.*)$', self._raw_control_data, re.M)
        return match.group(1)

    @property
    def architecture(self):
        match = re.search(r'^Architecture: (.*)$', self._raw_control_data, re.M)
        return match.group(1)

    @property
    def maintainer(self):
        match = re.search(r'^Maintainer: (.*)$', self._raw_control_data, re.M)
        return match.group(1)


class DebFile:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        logger.debug(f"Loading package {self.path}")
        self.ar_file = unix_ar.open(self.path)
        return self

    def __exit__(self, type, value, traceback):
        self.ar_file.close()
        logger.debug(f"Package {self.path} closed.")

    def control_file(self):
        # Loop over files in archive looking for the control archive.
        for name in map(lambda x: x.name, self.ar_file.infolist()):
            if name == b'control.tar' or name == b'control.tar.xz' or name == b'control.tar.gz':
                break
        else:
            logger.error("Control archive not found.")
            return None

        tarball = self.ar_file.open(name.decode())
        with tarfile.open(fileobj=tarball) as tar_file:
            member = tar_file.getmember('./control')
            content = tar_file.extractfile(member).read().decode()
        tarball.close()
        return ControlFile(content)

    def _is_elf_file(self, fileobj):
        magic = fileobj.read(4)
        fileobj.seek(0)
        return True if magic == b'\x7fELF' else False

    def _is_ar_file(self, fileobj):
        magic = fileobj.read(8)
        fileobj.seek(0)
        return True if magic == b'!<arch>\n' else False

    def iter_elffiles(self):
        # Loop over files in archive looking for the data archive
        for name in map(lambda x: x.name, self.ar_file.infolist()):
            if name.startswith(b"data.tar"):
                break
        else:
            logger.error("Data archive not found.")
            return
            
        tarball = self.ar_file.open(name.decode())
        with tarfile.open(fileobj=tarball) as tar_file:
            for member in filter(lambda x: x.isfile(), tar_file.getmembers()):
                file_obj = tar_file.extractfile(member)
                if self._is_elf_file(file_obj):
                    yield file_obj, os.path.basename(member.name)
                elif self._is_ar_file(file_obj):
                    yield from self._iter_elffiles_from_ar(unix_ar.open(file_obj), member)

        tarball.close()

    def _iter_elffiles_from_ar(self, ar_file, member):
        for ar_info in ar_file.infolist():
            fileobj = ar_file.open(ar_info.name.decode())
            if self._is_elf_file(fileobj):
                yield fileobj, f"{os.path.basename(member.name)}:{fileobj.name}"
