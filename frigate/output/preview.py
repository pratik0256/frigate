"""Handle outputting low res / fps preview segments from decoded frames."""

import logging
import multiprocessing as mp
import os
import subprocess as sp
import threading
from pathlib import Path

import cv2

from frigate.config import CameraConfig
from frigate.const import CACHE_DIR, CLIPS_DIR, INSERT_PREVIEW
from frigate.ffmpeg_presets import (
    EncodeTypeEnum,
    parse_preset_hardware_acceleration_encode,
)
from frigate.models import Previews

logger = logging.getLogger(__name__)

FOLDER_PREVIEW_FRAMES = "preview_frames"
SUMMARY_OUTPUT_FPS = 1
SUMMARY_SEGMENT_DURATION = 30


class FFMpegConverter(threading.Thread):
    """Convert a list of jpg frames into a vfr mp4."""

    def __init__(
        self,
        config: CameraConfig,
        frame_times: list[float],
        inter_process_queue: mp.Queue,
    ):
        threading.Thread.__init__(self)
        self.name = f"{config.name}_output_converter"
        self.camera = config.name
        self.frame_times = frame_times
        self.inter_process_queue = inter_process_queue
        self.path = os.path.join(
            CLIPS_DIR,
            f"previews/{self.camera}/{self.frame_times[0]}-{self.frame_times[-1]}.mp4",
        )

        # write a summary at fps and 1 key frame per clip
        self.ffmpeg_cmd = parse_preset_hardware_acceleration_encode(
            config.ffmpeg.hwaccel_args,
            input="-f concat -y -protocol_whitelist pipe,file -safe 0 -i /dev/stdin",
            output=f"-g {SUMMARY_OUTPUT_FPS * SUMMARY_SEGMENT_DURATION} -fpsmax {SUMMARY_OUTPUT_FPS} -bf 0 -b 9120 -fps_mode vfr -pix_fmt yuv420p {self.path}",
            type=EncodeTypeEnum.summary,
        )

    def run(self) -> None:
        # generate input list
        last = self.frame_times[0]
        playlist = []

        for t in self.frame_times:
            playlist.append(
                f"file '{os.path.join(CACHE_DIR, f'{FOLDER_PREVIEW_FRAMES}/preview_{self.camera}-{t}.jpg')}'"
            )
            playlist.append(f"duration {t - last}")
            last = t

        # last frame must be included again with no duration
        playlist.append(
            f"file '{os.path.join(CACHE_DIR, f'{FOLDER_PREVIEW_FRAMES}/preview_{self.camera}-{self.frame_times[-1]}.jpg')}'"
        )

        p = sp.run(
            self.ffmpeg_cmd.split(" "),
            input="\n".join(playlist),
            encoding="ascii",
            capture_output=True,
        )

        start = self.frame_times[0]
        end = self.frame_times[1]

        if p.returncode == 0:
            logger.info("successfully saved preview")
            self.inter_process_queue.put_nowait(
                (
                    INSERT_PREVIEW,
                    {
                        Previews.id: f"{end}-{start}",
                        Previews.camera: self.camera,
                        Previews.path: self.path,
                        Previews.start_time: start,
                        Previews.end_time: end,
                        Previews.duration: end - start,
                    },
                )
            )
        else:
            logger.error(f"Error saving preview for {self.camera} :: {p.stderr}")

        # unlink files from cache
        for t in self.frame_times:
            Path(
                os.path.join(
                    CACHE_DIR, f"{FOLDER_PREVIEW_FRAMES}/preview_{self.camera}-{t}.jpg"
                )
            ).unlink(missing_ok=True)


class PreviewRecorder:
    def __init__(self, config: CameraConfig, inter_process_queue: mp.Queue) -> None:
        self.config = config
        self.inter_process_queue = inter_process_queue
        self.start_time = 0
        self.last_output_time = 0
        self.output_frames = []
        self.out_height = 160
        self.out_width = int(
            (config.detect.width / config.detect.height) * self.out_height
        )

        Path(os.path.join(CACHE_DIR, "preview_frames")).mkdir(exist_ok=True)
        Path(os.path.join(CLIPS_DIR, f"previews/{config.name}")).mkdir(
            parents=True, exist_ok=True
        )

    def should_write_frame(
        self,
        current_tracked_objects: list[dict[str, any]],
        motion_boxes: list[list[int]],
        frame_time: float,
    ) -> bool:
        """Decide if this frame should be added to summary."""
        # limit output to 1 fps
        if (frame_time - self.last_output_time) < 1 / SUMMARY_OUTPUT_FPS:
            return False

        # send frame if a non-stationary object is in a zone
        if any(
            (len(o["current_zones"]) > 0 and not o["stationary"])
            for o in current_tracked_objects
        ):
            self.last_output_time = frame_time
            return True

        # TODO think of real motion box logic to use
        if len(motion_boxes) % 2 == 1:
            self.last_output_time = frame_time
            return True

        return False

    def write_frame_to_cache(self, frame_time: float, frame) -> None:
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_YUV2BGR_I420,
        )
        frame = cv2.resize(
            frame, dsize=(self.out_width, self.out_height), interpolation=cv2.INTER_AREA
        )
        _, jpg = cv2.imencode(".jpg", frame)
        with open(
            os.path.join(
                CACHE_DIR,
                f"{FOLDER_PREVIEW_FRAMES}/preview_{self.config.name}-{frame_time}.jpg",
            ),
            "wb",
        ) as j:
            j.write(jpg.tobytes())

    def write_data(
        self,
        current_tracked_objects: list[dict[str, any]],
        motion_boxes: list[list[int]],
        frame_time: float,
        frame,
    ) -> None:
        if self.start_time == 0:
            self.start_time = frame_time

        if self.should_write_frame(current_tracked_objects, motion_boxes, frame_time):
            self.output_frames.append(frame_time)
            self.write_frame_to_cache(frame_time, frame)

        # check if summary clip should be generated and cached frames reset
        if frame_time - self.start_time > SUMMARY_SEGMENT_DURATION:
            # save last frame to ensure consistent duration
            self.output_frames.append(frame_time)
            self.write_frame_to_cache(frame_time, frame)
            FFMpegConverter(
                self.config, self.output_frames, self.inter_process_queue
            ).start()

            # reset frame cache
            self.start_time = frame_time
            self.output_frames = []
