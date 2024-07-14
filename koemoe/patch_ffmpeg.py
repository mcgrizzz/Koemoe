import ffmpeg.asyncio
import ffprobe3.ffprobe3
import ffmpeg

from ffprobe3.ffprobe3 import _URI_SCHEME, _SPLIT_COMMAND_LINE, FFprobe
from ffprobe3.exceptions import *
from ffmpeg.utils import readlines

import os
import subprocess
import json

#money patch utf-8 encoding into ffprobe.probe
def probe(media_filename, *,
        communicate_timeout=10.0,  # a timeout in seconds
        ffprobe_cmd_override=None,
        verify_local_mediafile=True):

    split_cmdline = list(_SPLIT_COMMAND_LINE)  # Roger, copy that.
    ffprobe_cmd = split_cmdline[0]

    if ffprobe_cmd_override is not None:
        if not os.path.isfile(ffprobe_cmd_override):
            raise FFprobeOverrideFileError(ffprobe_cmd_override)
        else:
            ffprobe_cmd = ffprobe_cmd_override
            split_cmdline[0] = ffprobe_cmd_override

    if communicate_timeout is not None:
        # Verify that this non-None value is some kind of positive number.
        if not isinstance(communicate_timeout, (int, float)):
            raise FFprobeInvalidArgumentError('communicate_timeout',
                    'Supplied timeout is non-None and non-numeric',
                    communicate_timeout)
        if communicate_timeout <= 0.0:
            raise FFprobeInvalidArgumentError('communicate_timeout',
                    'Supplied timeout is non-None and non-positive',
                    communicate_timeout)

    if verify_local_mediafile:
        if _URI_SCHEME.match(media_filename) is None:
            # It doesn't look like the URI of a remote media file.
            if not os.path.isfile(media_filename):
                raise FFprobeMediaFileError(media_filename)

    split_cmdline.append(media_filename)

    try:
        proc = subprocess.Popen(split_cmdline,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
                universal_newlines=True)
    except FileNotFoundError as e:
        raise FFprobeExecutableError(ffprobe_cmd) from e
    except OSError as e:
        raise FFprobePopenError(e, 'OSError') from e
    except ValueError as e:
        raise FFprobePopenError(e, 'ValueError') from e
    except subprocess.SubprocessError as e:
        raise FFprobePopenError(e, 'subprocess.SubprocessError') from e

    try:
        (outs, errs) = proc.communicate(timeout=communicate_timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        (outs, errs) = proc.communicate()

    try:
        parsed_json = json.loads(outs)
    except json.decoder.JSONDecodeError as e:
        raise FFprobeJsonParseError(e, 'json.decoder.JSONDecodeError') from e
    exit_status = proc.returncode
    if exit_status != 0:
        raise FFprobeSubprocessError(split_cmdline, exit_status, errs)

    return FFprobe(split_cmdline=split_cmdline, parsed_json=parsed_json)

#Solution from: https://github.com/jonghwanhyeon/python-ffmpeg/pull/56
def _handle_stderr(self) -> str:
    assert self._process.stderr is not None
    line = ""
    for line_bytes in readlines(self._process.stderr):
        line = line_bytes.decode(errors="backslashreplace")
        self.emit("stderr", line)

    self._process.stderr.close()
    return line

async def _handle_stderr_async(self) -> str:
    assert self._process.stderr is not None

    line = ""
    async for line_bytes in readlines(self._process.stderr):
        line = line_bytes.decode(errors="backslashreplace")
        self.emit("stderr", line)

    return line

ffmpeg.FFmpeg._handle_stderr = _handle_stderr
ffmpeg.asyncio.FFmpeg._handle_stderr = _handle_stderr_async
ffprobe3.probe = probe