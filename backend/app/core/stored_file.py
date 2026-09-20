import mimetypes
from io import BytesIO

from starlette.datastructures import UploadFile as StarletteUploadFile


class StoredFile:
    """
    Единый объект для работы с файлами в проекте.
    Совместим с fastapi.UploadFile и boto3 upload_fileobj.
    """

    def __init__(
        self,
        filename: str = "",
        data: bytes | None = None,
        content_type: str | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.filename = filename
        self.headers = headers or {}
        self._content_type = content_type

        self._bytes_io = BytesIO(data) if data else BytesIO()
        self._bytes_io.seek(0)

    @property
    def content_type(self):
        if self._content_type is None:
            guessed_type, _ = mimetypes.guess_type(self.filename)
            self._content_type = guessed_type or "application/octet-stream"
        return self._content_type

    @property
    def file(self) -> BytesIO:
        """Возвращает BytesIO, перемотанный в начало — как у UploadFile.file"""
        self._bytes_io.seek(0)
        return self._bytes_io

    @property
    def size(self) -> int:
        current_pos = self._bytes_io.tell()
        self._bytes_io.seek(0, 2)
        size = self._bytes_io.tell()
        self._bytes_io.seek(current_pos)
        return size

    def as_upload_file(self) -> StarletteUploadFile:
        self.file.seek(0)
        return StarletteUploadFile(
            filename=self.filename,
            file=self.file,
            content_type=self.content_type,
            headers=self.headers,
        )

    def read(self) -> bytes:
        return self.file.read()

    def seek(self, offset: int):
        self.file.seek(offset)
