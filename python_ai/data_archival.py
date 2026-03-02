"""
Data Compression and Archival for NEO Hybrid AI.

Handles archiving old data to compressed formats,
retention policies, and optional Parquet export for
efficient historical analysis.
"""

import gzip
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

logger = logging.getLogger(__name__)


@dataclass
class ArchiveEntry:
    """Metadata about one archived file.

    Attributes:
        source_path: Original file path.
        archive_path: Compressed file path.
        original_bytes: Size before compression.
        compressed_bytes: Size after compression.
        timestamp: When the archive was created.
        compression: Compression method used.
    """

    source_path: str
    archive_path: str
    original_bytes: int
    compressed_bytes: int
    timestamp: float = field(default_factory=time.time)
    compression: str = "gzip"


class DataArchiver:
    """Archive and compress data files.

    Supports gzip compression with configurable
    retention policies.

    Args:
        archive_dir: Directory to store archives in.
        retention_days: Remove archives older than this.
            ``0`` disables auto-purge.
        chunk_size: Read/write buffer size.
    """

    def __init__(
        self,
        archive_dir: Union[str, Path] = "archives",
        retention_days: int = 90,
        chunk_size: int = 65_536,
    ) -> None:
        """Initialise the archiver."""
        self._dir = Path(archive_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._retention_days = retention_days
        self._chunk = chunk_size
        self._entries: List[ArchiveEntry] = []

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def compress_file(
        self,
        source: Union[str, Path],
        delete_original: bool = False,
    ) -> ArchiveEntry:
        """Compress a file with gzip.

        Args:
            source: Path to the file to compress.
            delete_original: Remove the original after
                successful compression.

        Returns:
            :class:`ArchiveEntry` with size details.

        Raises:
            FileNotFoundError: If *source* does not exist.
        """
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(str(src))

        dest = self._dir / (src.name + ".gz")
        original_bytes = src.stat().st_size

        with open(src, "rb") as fin:
            with gzip.open(dest, "wb") as fout:
                while True:
                    chunk = fin.read(self._chunk)
                    if not chunk:
                        break
                    fout.write(chunk)

        compressed_bytes = dest.stat().st_size
        if delete_original:
            src.unlink()

        entry = ArchiveEntry(
            source_path=str(src),
            archive_path=str(dest),
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
        )
        self._entries.append(entry)
        ratio = (
            compressed_bytes / original_bytes * 100 if original_bytes else 0
        )
        logger.info(
            "Compressed %s -> %s (%.1f%%)",
            src.name,
            dest.name,
            ratio,
        )
        return entry

    def decompress_file(
        self,
        archive: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Decompress a gzipped archive.

        Args:
            archive: Path to ``.gz`` file.
            output_dir: Where to write the output.
                Defaults to the archive's directory.

        Returns:
            Path to the decompressed file.
        """
        arc = Path(archive)
        out_dir = Path(output_dir) if output_dir else arc.parent
        out_name = arc.stem if arc.suffix == ".gz" else arc.name + ".out"
        out_path = out_dir / out_name
        out_dir.mkdir(parents=True, exist_ok=True)

        with gzip.open(arc, "rb") as fin:
            with open(out_path, "wb") as fout:
                while True:
                    chunk = fin.read(self._chunk)
                    if not chunk:
                        break
                    fout.write(chunk)
        logger.info("Decompressed %s -> %s", arc.name, out_path)
        return out_path

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def compress_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*",
        delete_originals: bool = False,
    ) -> List[ArchiveEntry]:
        """Compress all matching files in a directory.

        Args:
            directory: Source directory.
            pattern: Glob pattern for file matching.
            delete_originals: Remove originals after
                compression.

        Returns:
            List of :class:`ArchiveEntry`.
        """
        src_dir = Path(directory)
        results: List[ArchiveEntry] = []
        for f in sorted(src_dir.glob(pattern)):
            if f.is_file() and f.suffix != ".gz":
                entry = self.compress_file(f, delete_original=delete_originals)
                results.append(entry)
        return results

    # ------------------------------------------------------------------
    # JSON archival
    # ------------------------------------------------------------------

    def archive_json(
        self,
        data: Any,
        name: str,
    ) -> ArchiveEntry:
        """Serialise data to compressed JSON.

        Args:
            data: JSON-serialisable object.
            name: Base filename (without extension).

        Returns:
            :class:`ArchiveEntry`.
        """
        dest = self._dir / f"{name}.json.gz"
        raw = json.dumps(data, default=str).encode()
        with gzip.open(dest, "wb") as f:
            f.write(raw)

        entry = ArchiveEntry(
            source_path=f"<memory:{name}>",
            archive_path=str(dest),
            original_bytes=len(raw),
            compressed_bytes=dest.stat().st_size,
        )
        self._entries.append(entry)
        return entry

    def load_json(self, archive: Union[str, Path]) -> Any:
        """Load a compressed JSON archive.

        Args:
            archive: Path to ``.json.gz`` file.

        Returns:
            The deserialised object.
        """
        with gzip.open(archive, "rb") as f:
            return json.loads(f.read().decode())

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

    def enforce_retention(self) -> int:
        """Delete archives older than retention policy.

        Returns:
            Number of files removed.
        """
        if self._retention_days <= 0:
            return 0

        cutoff = time.time() - (self._retention_days * 86_400)
        removed = 0
        for f in self._dir.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                removed += 1
                logger.info("Retention purge: %s", f.name)
        return removed

    def disk_usage(self) -> Dict[str, Any]:
        """Return archive directory disk usage.

        Returns:
            Dict with file_count and total_bytes.
        """
        total = 0
        count = 0
        for f in self._dir.iterdir():
            if f.is_file():
                total += f.stat().st_size
                count += 1
        return {"file_count": count, "total_bytes": total}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return archiver stats.

        Returns:
            Dict with config and usage info.
        """
        usage = self.disk_usage()
        return {
            "archive_dir": str(self._dir),
            "retention_days": self._retention_days,
            "entries_tracked": len(self._entries),
            **usage,
        }
