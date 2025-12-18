#!/usr/bin/env python3
"""
Azure Video Indexer Processor
Process video and audio using Azure Video Indexer API
"""

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import moviepy.editor as mpy
import numpy as np
import requests
from utils.subtitle_utils import generate_srt_clip


class AzureVideoIndexerProcessor:
    """Processor for video/audio processing using Azure Video Indexer"""

    def __init__(
        self,
        subscription_key: str,
        location: str = "trial",
        account_id: Optional[str] = None,
        api_version: str = "2024-06-01-preview",
        bearer_token: Optional[str] = None,
        token_refresh_callback: Optional[Callable[[], Optional[str]]] = None,
    ):
        """
        Initialize Azure Video Indexer processor

        Args:
            subscription_key: Azure Video Indexer subscription key
            location: Azure region (default: "trial")
            account_id: Azure Video Indexer account ID (optional)
            api_version: API version
            bearer_token: Bearer token for authentication
            token_refresh_callback: Optional callback function to refresh token when expired.
                                   Should return a new bearer token string.
        """
        self.subscription_key = subscription_key
        self.location = location
        self.account_id = account_id
        self.api_version = api_version
        self.base_url = f"https://api.videoindexer.ai/{location}"
        # Use ARM data-plane bearer token (generateAccessToken) to avoid classic key flow.
        self.bearer_token = bearer_token
        self.token_refresh_callback = token_refresh_callback
        self.GLOBAL_COUNT = 0

        # If account_id is not provided, try to get it from API
        if not self.account_id:
            self.account_id = self._get_account_id()

        logging.info(
            f"Azure Video Indexer Processor initialized with account_id: {self.account_id}"
        )

    def _refresh_token(self) -> bool:
        """
        Attempt to refresh the bearer token using the callback.
        Returns True if token was refreshed successfully, False otherwise.
        """
        if not self.token_refresh_callback:
            logging.warning("Token expired but no refresh callback is configured")
            return False

        try:
            logging.info("Token expired, attempting to refresh...")
            new_token = self.token_refresh_callback()
            if new_token:
                self.bearer_token = new_token
                logging.info("Token refreshed successfully")
                return True
            else:
                logging.error("Token refresh callback returned empty token")
                return False
        except Exception as e:
            logging.error(f"Failed to refresh token: {e}")
            return False

    def _request_with_retry(
        self,
        method: str,
        url: str,
        max_retries: int = 1,
        **kwargs,
    ) -> requests.Response:
        """
        Make an HTTP request with automatic token refresh on 401 error.

        Args:
            method: HTTP method ('get', 'post', etc.)
            url: Request URL
            max_retries: Maximum number of retries after token refresh (default: 1)
            **kwargs: Additional arguments to pass to requests

        Returns:
            requests.Response object

        Raises:
            requests.HTTPError: If request fails after retries
        """
        request_func = getattr(requests, method.lower())

        for attempt in range(max_retries + 1):
            response = request_func(url, **kwargs)

            # If not 401, return immediately (success or other error)
            if response.status_code != 401:
                return response

            # 401 Unauthorized - try to refresh token
            if attempt < max_retries:
                logging.warning(f"Received 401 Unauthorized (attempt {attempt + 1})")
                if self._refresh_token():
                    # Update auth params with new token
                    if "params" in kwargs and "accessToken" in kwargs["params"]:
                        kwargs["params"]["accessToken"] = self.bearer_token
                    logging.info("Retrying request with new token...")
                else:
                    # Can't refresh, return the 401 response
                    return response
            else:
                logging.error("Max retries reached, returning 401 response")

        return response

    def _get_account_id(self) -> str:
        """Get account ID using either subscription key or bearer token"""
        url = f"{self.base_url}/Accounts"

        # Use bearer token if available (ARM flow), otherwise use subscription key (Classic flow)
        if self.bearer_token:
            headers = {"Authorization": f"Bearer {self.bearer_token}"}
            logging.info("Using bearer token to retrieve account_id")
        elif self.subscription_key:
            headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
            logging.info("Using subscription key to retrieve account_id")
        else:
            raise ValueError(
                "Cannot retrieve account_id: neither bearer_token nor subscription_key is provided. "
                "Please set AZURE_VIDEO_INDEXER_ACCOUNT_ID in .env"
            )

        response = requests.get(url, headers=headers)
        response.raise_for_status()
        accounts = response.json()
        if accounts:
            return accounts[0]["id"]
        raise ValueError(
            "No account found. Please provide account_id explicitly via AZURE_VIDEO_INDEXER_ACCOUNT_ID in .env"
        )

    def _get_access_token(self) -> str:
        """
        Return data-plane access token.
        Requires bearer_token pre-generated via ARM generateAccessToken API.
        """
        if not self.bearer_token:
            raise ValueError(
                "bearer_token is required. Generate it via ARM generateAccessToken and pass it to the processor."
            )
        return self.bearer_token

    def _auth(self, access_token: str) -> Tuple[Dict, Dict]:
        """
        Build (params, headers) for bearer mode.
        For ARM-based accounts, token should be in URL params.
        """
        return {"accessToken": access_token}, {}

    def upload_video(
        self, video_path: str, video_name: Optional[str] = None, language: str = "en-US"
    ) -> str:
        """
        Upload video to Azure Video Indexer

        Args:
            video_path: Video file path
            video_name: Video name (optional)
            language: Video language (default: en-US)

        Returns:
            Video ID
        """
        access_token = self._get_access_token()
        params_auth, headers_auth = self._auth(access_token)
        url = f"{self.base_url}/Accounts/{self.account_id}/Videos"

        if not video_name:
            video_name = os.path.basename(video_path)

        # Get file size
        file_size = os.path.getsize(video_path)
        file_size_mb = file_size / (1024 * 1024)

        logging.info("=" * 60)
        logging.info("Uploading video to Azure Video Indexer")
        logging.info(f"File name: {video_name}")
        logging.info(f"File size: {file_size_mb:.2f} MB")
        logging.info(f"Language: {language}")
        logging.info("=" * 60)

        params = {
            "name": video_name,
            "language": language,
            "privacy": "Private",
        }
        params.update(params_auth)

        # Determine MIME type based on file extension
        ext = os.path.splitext(video_path)[1].lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
            ".mkv": "video/x-matroska",
            ".wmv": "video/x-ms-wmv",
            ".webm": "video/webm",
            ".m4v": "video/x-m4v",
            ".flv": "video/x-flv",
        }
        mime_type = mime_types.get(ext, "video/mp4")
        logging.info(f"File format: {ext}, MIME type: {mime_type}")

        logging.info("Uploading file...")

        # Upload with retry on 401 (token expired)
        max_retries = 1
        for attempt in range(max_retries + 1):
            with open(video_path, "rb") as video_file:
                files = {"file": (video_name, video_file, mime_type)}
                response = requests.post(url, params=params, files=files, headers=headers_auth)

            if response.status_code == 401 and attempt < max_retries:
                logging.warning("Upload failed with 401, attempting token refresh...")
                if self._refresh_token():
                    # Update params with new token
                    access_token = self._get_access_token()
                    params_auth, headers_auth = self._auth(access_token)
                    params["accessToken"] = access_token
                    logging.info("Retrying upload with new token...")
                    continue
            break

        response.raise_for_status()
        result = response.json()
        video_id = result["id"]

        logging.info("Upload successful!")
        logging.info(f"Video ID: {video_id}")
        logging.info(f"Initial state: {result.get('state', 'Unknown')}")
        logging.info("=" * 60)

        return video_id

    def upload_audio(
        self, audio_path: str, audio_name: Optional[str] = None, language: str = "en-US"
    ) -> str:
        """
        Upload audio to Azure Video Indexer

        Args:
            audio_path: Audio file path
            audio_name: Audio name (optional)
            language: Audio language (default: en-US)

        Returns:
            Video ID (Azure Video Indexer processes audio as video)
        """
        access_token = self._get_access_token()
        params_auth, headers_auth = self._auth(access_token)
        url = f"{self.base_url}/Accounts/{self.account_id}/Videos"

        if not audio_name:
            audio_name = os.path.basename(audio_path)

        # Get file size
        file_size = os.path.getsize(audio_path)
        file_size_mb = file_size / (1024 * 1024)

        logging.info("=" * 60)
        logging.info("Uploading audio to Azure Video Indexer")
        logging.info(f"File name: {audio_name}")
        logging.info(f"File size: {file_size_mb:.2f} MB")
        logging.info(f"Language: {language}")
        logging.info("=" * 60)

        params = {
            "name": audio_name,
            "language": language,
            "privacy": "Private",
        }
        params.update(params_auth)

        # Determine MIME type based on file extension
        ext = os.path.splitext(audio_path)[1].lower()
        mime_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".aac": "audio/aac",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".wma": "audio/x-ms-wma",
            ".m4a": "audio/mp4",
        }
        mime_type = mime_types.get(ext, "audio/wav")
        logging.info(f"File format: {ext}, MIME type: {mime_type}")

        logging.info("Uploading file...")

        # Upload with retry on 401 (token expired)
        max_retries = 1
        for attempt in range(max_retries + 1):
            with open(audio_path, "rb") as audio_file:
                files = {"file": (audio_name, audio_file, mime_type)}
                response = requests.post(url, params=params, files=files, headers=headers_auth)

            if response.status_code == 401 and attempt < max_retries:
                logging.warning("Upload failed with 401, attempting token refresh...")
                if self._refresh_token():
                    # Update params with new token
                    access_token = self._get_access_token()
                    params_auth, headers_auth = self._auth(access_token)
                    params["accessToken"] = access_token
                    logging.info("Retrying upload with new token...")
                    continue
            break

        response.raise_for_status()
        result = response.json()
        video_id = result["id"]

        logging.info("Upload successful!")
        logging.info(f"Video ID: {video_id}")
        logging.info(f"Initial state: {result.get('state', 'Unknown')}")
        logging.info("=" * 60)

        return video_id

    def wait_for_indexing(self, video_id: str, timeout: int = 600) -> Dict:
        """
        Wait for video indexing to complete

        Args:
            video_id: Video ID
            timeout: Timeout in seconds (default: 10 minutes)

        Returns:
            Indexing result
        """
        url = f"{self.base_url}/Accounts/{self.account_id}/Videos/{video_id}/Index"

        logging.info(f"Waiting for video processing... Video ID: {video_id}")
        logging.info(f"Maximum wait time: {timeout} seconds ({timeout / 60:.1f} minutes)")

        start_time = time.time()
        check_count = 0

        while time.time() - start_time < timeout:
            check_count += 1
            elapsed = time.time() - start_time

            # Get fresh token for each request (handles token refresh)
            access_token = self._get_access_token()
            params_auth, headers_auth = self._auth(access_token)
            params = {}
            params.update(params_auth)

            response = self._request_with_retry("get", url, params=params, headers=headers_auth)
            response.raise_for_status()
            result = response.json()

            state = result.get("state", "").lower()

            # Get processing progress
            progress = "Unknown"
            if "videos" in result and len(result["videos"]) > 0:
                progress = result["videos"][0].get("processingProgress", "Unknown")

            duration = result.get("durationInSeconds", 0)

            if state == "processed":
                logging.info("=" * 60)
                logging.info("Video processing completed successfully!")
                logging.info(f"Video duration: {duration} seconds")
                logging.info(
                    f"Total time elapsed: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)"
                )
                logging.info(f"Check count: {check_count}")
                logging.info("=" * 60)
                return result

            elif state == "failed":
                error_msg = result.get("error", "Unknown error")
                failure_msg = result.get("videos", [{}])[0].get("failureMessage", "")
                logging.error(f"Video processing failed: {error_msg}")
                if failure_msg:
                    logging.error(f"Failure details: {failure_msg}")
                raise RuntimeError(f"Video indexing failed: {error_msg}")

            # Detailed processing status log
            logging.info(
                f"[{check_count}] Processing... "
                f"State: {state} | "
                f"Progress: {progress} | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Duration: {duration}s"
            )
            time.sleep(5)

        raise TimeoutError(f"Video indexing timeout after {timeout} seconds")

    def get_transcript(self, video_id: str) -> Tuple[str, str, Dict]:
        """
        Get video transcription results

        Args:
            video_id: Video ID

        Returns:
            (transcript text, SRT subtitles, state dictionary)
        """
        access_token = self._get_access_token()
        params_auth, headers_auth = self._auth(access_token)

        # Get indexing results
        url = f"{self.base_url}/Accounts/{self.account_id}/Videos/{video_id}/Index"
        params = {}
        params.update(params_auth)
        response = self._request_with_retry("get", url, params=params, headers=headers_auth)
        response.raise_for_status()
        index_result = response.json()

        # Save API response to file for debugging
        import json

        debug_output_path = os.path.join(os.path.dirname(__file__), "..", "api_response.json")
        try:
            with open(debug_output_path, "w", encoding="utf-8") as f:
                json.dump(index_result, f, indent=2, ensure_ascii=False)
            logging.info(f"API response saved to: {os.path.abspath(debug_output_path)}")
        except Exception as e:
            logging.warning(f"Failed to save API response: {e}")

        # Extract transcript text and subtitles
        transcript_text = ""
        srt_content = ""
        sentences = []

        # Extract transcript from insights
        insights = index_result.get("videos", [{}])[0].get("insights", {})
        transcript_items = insights.get("transcript", [])

        # Log first transcript item for quick debugging
        if transcript_items:
            first_item = transcript_items[0]
            logging.info(f"First transcript item: {json.dumps(first_item, indent=2)}")

        if transcript_items:
            # Build complete text
            transcript_text = " ".join([item.get("text", "") for item in transcript_items])

            # Build SRT subtitles and sentence information (compatible with original format)
            srt_lines = []
            for idx, item in enumerate(transcript_items, 1):
                text = item.get("text", "").strip()

                # Get timestamps from instances array (Azure Video Indexer format)
                instances = item.get("instances", [{}])
                if instances:
                    instance = instances[0]
                    start_str = instance.get("start", "0:00:00")
                    end_str = instance.get("end", "0:00:00")
                else:
                    start_str = "0:00:00"
                    end_str = "0:00:00"

                # Parse Azure timestamp strings to seconds
                start_seconds = self._parse_azure_timestamp(start_str)
                end_seconds = self._parse_azure_timestamp(end_str)

                # Format to SRT timestamp
                start_time = self._format_timestamp(start_seconds)
                end_time = self._format_timestamp(end_seconds)

                srt_lines.append(f"{idx}")
                srt_lines.append(f"{start_time} --> {end_time}")
                srt_lines.append(text)
                srt_lines.append("")

                # Build sentence information (compatible with original format)
                # Original format requires timestamp to be a list in milliseconds
                start_ms = int(start_seconds * 1000)
                end_ms = int(end_seconds * 1000)
                timestamp = [[start_ms, end_ms]]

                sentences.append(
                    {
                        "text": text,
                        "timestamp": timestamp,  # Format: [[start_ms, end_ms], ...]
                        "start": start_seconds,
                        "end": end_seconds,
                    }
                )

            srt_content = "\n".join(srt_lines)

        # Build state dictionary (compatible with original interface)
        state = {
            "video_id": video_id,
            "transcript_text": transcript_text,
            "sentences": sentences,
            "index_result": index_result,
        }

        return transcript_text, srt_content, state

    def _parse_azure_timestamp(self, timestamp_str: str) -> float:
        """Parse Azure Video Indexer timestamp string to seconds.

        Azure format: "0:00:00.04" or "0:05:00.2919999"
        """
        if isinstance(timestamp_str, (int, float)):
            return float(timestamp_str)

        try:
            parts = timestamp_str.split(":")
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            else:
                return float(timestamp_str)
        except (ValueError, AttributeError):
            return 0.0

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp to SRT format (00:00:00,000)"""
        if isinstance(seconds, str):
            seconds = self._parse_azure_timestamp(seconds)
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def recog(
        self,
        audio_input: Tuple[int, np.ndarray],
        sd_switch: str = "no",
        state: Optional[Dict] = None,
        hotwords: str = "",
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str, Dict]:
        """
        Recognize audio (compatible with original interface)

        Args:
            audio_input: (sample rate, audio data)
            sd_switch: Speaker diarization switch (Azure Video Indexer automatically supports)
            state: State dictionary
            hotwords: Hotwords (Azure Video Indexer doesn't support, kept for interface compatibility)
            output_dir: Output directory

        Returns:
            (transcript text, SRT subtitles, state dictionary)
        """
        if state is None:
            state = {}

        # Save audio to temporary file
        import tempfile

        import soundfile as sf

        sr, data = audio_input
        temp_audio_path = None

        try:
            temp_audio_path = tempfile.mktemp(suffix=".wav")
            sf.write(temp_audio_path, data, sr)

            # Upload and process audio
            video_id = self.upload_audio(temp_audio_path, language="en-US")
            self.wait_for_indexing(video_id)
            transcript_text, srt_content, result_state = self.get_transcript(video_id)

            # Merge state
            state.update(result_state)
            state["audio_input"] = audio_input

            return transcript_text, srt_content, state
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

    def video_recog(
        self,
        video_filename: str,
        sd_switch: str = "no",
        hotwords: str = "",
        output_dir: Optional[str] = None,
    ) -> Tuple[str, str, Dict]:
        """
        Recognize video (compatible with original interface)

        Args:
            video_filename: Video file path
            sd_switch: Speaker diarization switch (Azure Video Indexer automatically supports)
            hotwords: Hotwords (Azure Video Indexer doesn't support, kept for interface compatibility)
            output_dir: Output directory

        Returns:
            (transcript text, SRT subtitles, state dictionary)
        """
        state = {}

        # Upload and process video
        video_id = self.upload_video(video_filename, language="en-US")
        self.wait_for_indexing(video_id)
        transcript_text, srt_content, result_state = self.get_transcript(video_id)

        # Merge state
        state.update(result_state)
        state["video_filename"] = video_filename

        # Generate output filename
        base_name, _ = os.path.splitext(video_filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            _, base_name = os.path.split(video_filename)
            base_name, _ = os.path.splitext(base_name)
        clip_video_file = base_name + "_clip.mp4"
        if output_dir:
            clip_video_file = os.path.join(output_dir, clip_video_file)
        state["clip_video_file"] = clip_video_file

        return transcript_text, srt_content, state

    def clip(
        self,
        dest_text: str,
        start_ost: int,
        end_ost: int,
        state: Dict,
        dest_spk: Optional[str] = None,
        output_dir: Optional[str] = None,
        timestamp_list: Optional[List] = None,
    ) -> Tuple[Tuple[int, np.ndarray], str, str]:
        """
        Clip audio (compatible with original interface)

        Args:
            dest_text: Target text
            start_ost: Start offset (milliseconds)
            end_ost: End offset (milliseconds)
            state: State dictionary
            dest_spk: Target speaker (Azure Video Indexer supports this, but needs to be extracted from insights)
            output_dir: Output directory
            timestamp_list: Timestamp list (optional)

        Returns:
            ((sample rate, audio data), message, SRT subtitles)
        """
        # Get audio input from state
        audio_input = state.get("audio_input")
        if not audio_input:
            raise ValueError("Audio input not found in state")

        sr, data = audio_input
        sentences = state.get("sentences", [])

        # Find matching timestamps
        all_ts = []
        if timestamp_list:
            all_ts = [[t[0] * 16, t[1] * 16] for t in timestamp_list]
        else:
            # Find matching text from sentences
            for _dest_text in dest_text.split("#"):
                _dest_text = _dest_text.strip()
                for sentence in sentences:
                    if _dest_text.lower() in sentence.get("text", "").lower():
                        # Use first and last timestamps from timestamp
                        timestamp = sentence.get("timestamp", [])
                        if timestamp:
                            start_ms = (
                                timestamp[0][0] * 16 / 1000
                            )  # Convert milliseconds to samples
                            end_ms = timestamp[-1][1] * 16 / 1000
                            all_ts.append([int(start_ms), int(end_ms)])
                        else:
                            # Fallback to start/end
                            start_ms = sentence.get("start", 0) * sr  # Convert seconds to samples
                            end_ms = sentence.get("end", 0) * sr
                            all_ts.append([int(start_ms), int(end_ms)])

        # Perform clipping
        clip_srt = ""
        if all_ts:
            start, end = all_ts[0]
            start = min(max(0, start + start_ost * 16), len(data))
            end = min(max(0, end + end_ost * 16), len(data))
            res_audio = data[start:end]
            start_end_info = f"from {start / sr} to {end / sr}"

            # Generate clipped SRT
            srt_clip, _, _ = generate_srt_clip(sentences, start / sr, end / sr, begin_index=0)
            clip_srt += srt_clip

            # Process multiple time periods
            for _ts in all_ts[1:]:
                start, end = _ts
                start = min(max(0, start + start_ost * 16), len(data))
                end = min(max(0, end + end_ost * 16), len(data))
                start_end_info += f", from {start / sr} to {end / sr}"
                res_audio = np.concatenate([res_audio, data[start:end]], -1)
                srt_clip, _, _ = generate_srt_clip(sentences, start / sr, end / sr, begin_index=0)
                clip_srt += srt_clip

            message = f"{len(all_ts)} periods found in the speech: " + start_end_info
        else:
            message = "No period found in the speech, return raw speech."
            res_audio = data

        return (sr, res_audio), message, clip_srt

    def video_clip(
        self,
        dest_text: str,
        start_ost: int,
        end_ost: int,
        state: Dict,
        font_size: int = 32,
        font_color: str = "white",
        add_sub: bool = False,
        dest_spk: Optional[str] = None,
        output_dir: Optional[str] = None,
        timestamp_list: Optional[List] = None,
    ) -> Tuple[str, str, str]:
        """
        Clip video (compatible with original interface)

        Args:
            dest_text: Target text
            start_ost: Start offset (milliseconds)
            end_ost: End offset (milliseconds)
            state: State dictionary
            font_size: Subtitle font size
            font_color: Subtitle color
            add_sub: Whether to add subtitles
            dest_spk: Target speaker
            output_dir: Output directory
            timestamp_list: Timestamp list (optional)

        Returns:
            (video file path, message, SRT subtitles)
        """
        video_filename = state.get("video_filename")
        if not video_filename:
            raise ValueError("Video filename not found in state")

        sentences = state.get("sentences", [])
        clip_video_file = state.get("clip_video_file", "output_clip.mp4")

        # Find matching timestamps
        all_ts = []
        if timestamp_list:
            all_ts = [[t[0], t[1]] for t in timestamp_list]
        else:
            # Find matching text from sentences
            for _dest_text in dest_text.split("#"):
                _dest_text = _dest_text.strip()
                for sentence in sentences:
                    if _dest_text.lower() in sentence.get("text", "").lower():
                        # Use first and last timestamps from timestamp (convert to seconds)
                        timestamp = sentence.get("timestamp", [])
                        if timestamp:
                            start = timestamp[0][0] / 1000.0  # Convert milliseconds to seconds
                            end = timestamp[-1][1] / 1000.0
                        else:
                            start = sentence.get("start", 0)
                            end = sentence.get("end", 0)
                        all_ts.append([start, end])

        clip_srt = ""
        if all_ts:
            video = None
            clips_to_close = []
            try:
                video = mpy.VideoFileClip(video_filename)
                clips_to_close.append(video)

                if add_sub:
                    from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
                    from moviepy.video.tools.subtitles import SubtitlesClip, TextClip

                    def generator(txt):
                        return TextClip(
                            txt,
                            fontsize=font_size,
                            color=font_color,
                        )

                concate_clip = []
                start_end_info = ""

                # Build clips for each time period
                for i, _ts in enumerate(all_ts):
                    start, end = _ts
                    start = start + start_ost / 1000.0
                    end = end + end_ost / 1000.0

                    _clip = video.subclip(start, end)
                    clips_to_close.append(_clip)

                    if i == 0:
                        start_end_info = f"from {start} to {end}"
                    else:
                        start_end_info += f", from {start:.2f} to {end:.2f}"

                    srt_clip, subs, _ = generate_srt_clip(sentences, start, end, begin_index=0)
                    clip_srt += srt_clip

                    if add_sub:
                        subtitles = SubtitlesClip(subs, generator)
                        _clip = CompositeVideoClip(
                            [_clip, subtitles.set_pos(("center", "bottom"))]  # type: ignore
                        )
                        clips_to_close.append(_clip)

                    concate_clip.append(_clip)

                # Concatenate video clips
                final_clip = concate_clip[0]
                if len(concate_clip) > 1:
                    from moviepy.editor import concatenate_videoclips

                    final_clip = concatenate_videoclips(concate_clip)
                    clips_to_close.append(final_clip)

                # Save video
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    _, file_with_extension = os.path.split(clip_video_file)
                    clip_video_file_name, _ = os.path.splitext(file_with_extension)
                    clip_video_file = os.path.join(
                        output_dir, f"{clip_video_file_name}_no{self.GLOBAL_COUNT}.mp4"
                    )
                else:
                    clip_video_file = clip_video_file[:-4] + f"_no{self.GLOBAL_COUNT}.mp4"

                final_clip.write_videofile(
                    clip_video_file, audio_codec="aac", verbose=False, logger=None
                )
                self.GLOBAL_COUNT += 1

                message = f"{len(all_ts)} periods found in the video: " + start_end_info
            finally:
                # Close MoviePy clips to release file locks on Windows
                seen = set()
                for c in reversed(clips_to_close):
                    try:
                        if c and id(c) not in seen:
                            seen.add(id(c))
                            c.close()
                    except Exception:
                        logging.warning("Failed to close video clip", exc_info=True)
        else:
            clip_video_file = state.get("video_filename", "output.mp4")
            message = "No period found in the video, return raw video."
            clip_srt = ""

        return clip_video_file, message, clip_srt

    def video_clip_segments(
        self,
        state: Dict,
        output_dir: Optional[str] = None,
        timestamp_list: Optional[List] = None,
        start_ost: int = 0,
        end_ost: int = 100,
    ) -> List[Dict]:
        """
        Clip video into multiple separate segments based on timestamp_list.

        Args:
            state: State dictionary containing video_filename and sentences
            output_dir: Output directory for clipped videos
            timestamp_list: List of [start_ms, end_ms] timestamps
            start_ost: Start offset in milliseconds
            end_ost: End offset in milliseconds

        Returns:
            List of dictionaries, each containing:
                - 'video_path': Path to the clipped video file
                - 'transcript': Transcript text for this segment
                - 'srt': SRT subtitle for this segment
                - 'start': Start time in seconds
                - 'end': End time in seconds
        """
        video_filename = state.get("video_filename")
        if not video_filename:
            raise ValueError("Video filename not found in state")

        sentences = state.get("sentences", [])
        segments = []

        if not timestamp_list:
            return segments

        import subprocess

        # Generate base output filename
        base_name, _ = os.path.splitext(os.path.basename(video_filename))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for i, ts in enumerate(timestamp_list):
            start_sec = ts[0] / 1000.0 + start_ost / 1000.0
            end_sec = ts[1] / 1000.0 + end_ost / 1000.0
            duration = end_sec - start_sec

            # Ensure valid time range
            start_sec = max(0, start_sec)
            if duration <= 0:
                continue

            # Generate output filename
            segment_filename = f"{base_name}_segment_{i + 1}.mp4"
            if output_dir:
                segment_path = os.path.join(output_dir, segment_filename)
            else:
                segment_path = os.path.join(
                    os.path.dirname(video_filename) or ".", segment_filename
                )

            logging.info(
                f"Clipping segment {i + 1}: {start_sec:.2f}s - {end_sec:.2f}s -> {segment_path}"
            )

            # Use ffmpeg directly for reliable video clipping
            # -ss before -i for fast seeking, -t for duration
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output
                "-ss",
                str(start_sec),  # Start time
                "-i",
                video_filename,  # Input file
                "-t",
                str(duration),  # Duration
                "-c:v",
                "libx264",  # Video codec
                "-c:a",
                "aac",  # Audio codec
                "-preset",
                "fast",  # Encoding speed
                "-avoid_negative_ts",
                "make_zero",  # Fix timestamp issues
                segment_path,
            ]

            try:
                subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                logging.info(f"Segment {i + 1} saved successfully")
            except subprocess.CalledProcessError as e:
                logging.error(f"FFmpeg error for segment {i + 1}: {e.stderr}")
                raise RuntimeError(f"FFmpeg failed: {e.stderr}") from e

            # Generate transcript and SRT for this segment
            srt_clip, _, _ = generate_srt_clip(sentences, start_sec, end_sec, begin_index=0)

            # Extract transcript text for this time range
            transcript_text = ""
            for sentence in sentences:
                sent_start = sentence.get("start", 0)
                sent_end = sentence.get("end", 0)
                # Check if sentence overlaps with segment time range
                if sent_start < end_sec and sent_end > start_sec:
                    transcript_text += sentence.get("text", "") + " "

            segments.append(
                {
                    "video_path": segment_path,
                    "transcript": transcript_text.strip(),
                    "srt": srt_clip,
                    "start": start_sec,
                    "end": end_sec,
                    "index": i + 1,
                }
            )

        return segments

    def audio_clip_segments(
        self,
        state: Dict,
        output_dir: Optional[str] = None,
        timestamp_list: Optional[List] = None,
        start_ost: int = 0,
        end_ost: int = 100,
    ) -> List[Dict]:
        """
        Clip audio into multiple separate segments based on timestamp_list.

        Args:
            state: State dictionary containing audio_input and sentences
            output_dir: Output directory for clipped audio
            timestamp_list: List of [start_ms, end_ms] timestamps
            start_ost: Start offset in milliseconds
            end_ost: End offset in milliseconds

        Returns:
            List of dictionaries, each containing:
                - 'audio_data': Tuple of (sample_rate, audio_array)
                - 'transcript': Transcript text for this segment
                - 'srt': SRT subtitle for this segment
                - 'start': Start time in seconds
                - 'end': End time in seconds
        """
        audio_input = state.get("audio_input")
        if not audio_input:
            raise ValueError("Audio input not found in state")

        sr, data = audio_input
        sentences = state.get("sentences", [])
        segments = []

        if not timestamp_list:
            return segments

        for i, ts in enumerate(timestamp_list):
            start_sec = ts[0] / 1000.0 + start_ost / 1000.0
            end_sec = ts[1] / 1000.0 + end_ost / 1000.0

            # Convert to samples
            start_sample = int(start_sec * sr)
            end_sample = int(end_sec * sr)

            # Ensure valid range
            start_sample = max(0, start_sample)
            end_sample = min(len(data), end_sample)

            if end_sample <= start_sample:
                continue

            # Extract audio segment
            audio_segment = data[start_sample:end_sample]

            # Generate transcript and SRT for this segment
            srt_clip, _, _ = generate_srt_clip(sentences, start_sec, end_sec, begin_index=0)

            # Extract transcript text for this time range
            transcript_text = ""
            for sentence in sentences:
                sent_start = sentence.get("start", 0)
                sent_end = sentence.get("end", 0)
                # Check if sentence overlaps with segment time range
                if sent_start < end_sec and sent_end > start_sec:
                    transcript_text += sentence.get("text", "") + " "

            segments.append(
                {
                    "audio_data": (sr, audio_segment),
                    "transcript": transcript_text.strip(),
                    "srt": srt_clip,
                    "start": start_sec,
                    "end": end_sec,
                    "index": i + 1,
                }
            )

        return segments
