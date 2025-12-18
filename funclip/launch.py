import argparse
import logging
import os

import gradio as gr
import requests
from azure_processor import AzureVideoIndexerProcessor
from introduction import top_md_1, top_md_3, top_md_4
from llm.gemini_api import gemini_call
from utils.trans_utils import extract_timestamps_with_text

# Set custom temp directory for Gradio to avoid Windows permission issues
# Use project directory instead of system temp to avoid antivirus interference
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRADIO_TEMP_DIR = os.path.join(SCRIPT_DIR, ".gradio_temp")
os.makedirs(GRADIO_TEMP_DIR, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = GRADIO_TEMP_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
    logging.info("Environment variables loaded from .env file")
except ImportError:
    logging.warning(
        "python-dotenv not installed. Reading from system environment only. Install with: pip install python-dotenv"
    )

# Global MAX_SEGMENTS configuration - read from environment variable
# This controls how many video segments can be displayed in the UI
# Must be defined AFTER load_dotenv() to read from .env file
try:
    MAX_SEGMENTS = int(os.getenv("MAX_SEGMENTS", "10"))
except ValueError:
    logging.warning("Invalid MAX_SEGMENTS value in environment, using default (10)")
    MAX_SEGMENTS = 10

# Authentication - read from environment variables
# If both USERNAME and PASSWORD are set, users must enter them to access the app
APP_USERNAME = os.getenv("USERNAME", "")
APP_PASSWORD = os.getenv("PASSWORD", "")


def generate_arm_token(sub_id, rg, account):
    """
    Generate data-plane access token using Azure Identity (ARM flow).
    Supports multiple authentication methods via environment variables:
    1. Service Principal: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID
    2. Interactive Browser: Set USE_INTERACTIVE_AUTH=true
    3. Device Code: Set USE_DEVICE_CODE_AUTH=true
    4. Default chain (CLI, Managed Identity, etc.)
    """
    try:
        from azure.identity import (
            ClientSecretCredential,
            DefaultAzureCredential,
            DeviceCodeCredential,
            InteractiveBrowserCredential,
        )
    except ImportError as err:
        logging.error("azure-identity not found.")
        raise ImportError(
            "Production-grade token generation requires 'azure-identity'. "
            "Please install it: pip install azure-identity"
        ) from err

    logging.info("Generating access token using Azure Identity for account: %s", account)

    try:
        # Determine which credential to use based on environment variables
        client_id = os.getenv("AZURE_CLIENT_ID")
        client_secret = os.getenv("AZURE_CLIENT_SECRET")
        tenant_id = os.getenv("AZURE_TENANT_ID")
        use_interactive = os.getenv("USE_INTERACTIVE_AUTH", "").lower().strip()
        use_device_code = os.getenv("USE_DEVICE_CODE_AUTH", "").lower().strip()

        # Debug logging
        logging.info("=== Authentication Configuration Debug ===")
        logging.info(f"AZURE_CLIENT_ID set: {bool(client_id)}")
        logging.info(f"AZURE_CLIENT_SECRET set: {bool(client_secret)}")
        logging.info(f"AZURE_TENANT_ID set: {bool(tenant_id)}")
        logging.info(f"USE_INTERACTIVE_AUTH: '{use_interactive}'")
        logging.info(f"USE_DEVICE_CODE_AUTH: '{use_device_code}'")
        logging.info("=========================================")

        # Priority 1: Service Principal (if all three are set)
        if client_id and client_secret and tenant_id:
            logging.info("✓ Using Service Principal authentication")
            cred = ClientSecretCredential(
                tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
            )
        # Priority 2: Interactive Browser
        elif use_interactive == "true":
            logging.info("✓ Using Interactive Browser authentication")
            cred = (
                InteractiveBrowserCredential(tenant_id=tenant_id)
                if tenant_id
                else InteractiveBrowserCredential()
            )
        # Priority 3: Device Code
        elif use_device_code == "true":
            logging.info("✓ Using Device Code authentication")
            cred = (
                DeviceCodeCredential(tenant_id=tenant_id) if tenant_id else DeviceCodeCredential()
            )
        # Priority 4: Default credential chain
        else:
            logging.warning(
                "⚠ No explicit auth method configured, falling back to DefaultAzureCredential"
            )
            logging.warning("⚠ Recommendation: Add 'USE_INTERACTIVE_AUTH=true' to your .env file")
            cred = DefaultAzureCredential()

        token_obj = cred.get_token("https://management.azure.com/.default")
        mgmt_token = token_obj.token

        uri = f"https://management.azure.com/subscriptions/{sub_id}/resourceGroups/{rg}/providers/Microsoft.VideoIndexer/accounts/{account}/generateAccessToken?api-version=2024-01-01"

        headers = {
            "Authorization": f"Bearer {mgmt_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "permissionType": "Contributor",
            "scope": "Account",
        }

        response = requests.post(uri, headers=headers, json=payload)
        response.raise_for_status()

        return response.json()["accessToken"]

    except Exception as e:
        raise RuntimeError(f"Failed to generate ARM token: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoClipper with Azure Video Indexer (reads config from .env by default)"
    )
    parser.add_argument("--share", "-s", action="store_true", help="establish gradio share link")
    parser.add_argument(
        "--port", "-p", type=int, help="port number (default: 7860 or GRADIO_PORT from env)"
    )
    parser.add_argument("--listen", action="store_true", help="listen to all hosts")

    args = parser.parse_args()

    # Read all configuration from environment variables
    # Authentication: Bearer Token (preferred)
    bearer_token = os.getenv("AZURE_VIDEO_INDEXER_BEARER_TOKEN")

    # ARM auto-generation parameters (if bearer token not provided)
    arm_subscription_id = os.getenv("ARM_SUBSCRIPTION_ID")
    arm_resource_group = os.getenv("ARM_RESOURCE_GROUP")
    arm_account_name = os.getenv("ARM_ACCOUNT_NAME")

    # Classic authentication (fallback)
    subscription_key = os.getenv("AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY")
    account_id = os.getenv("AZURE_VIDEO_INDEXER_ACCOUNT_ID")
    location = os.getenv("AZURE_VIDEO_INDEXER_LOCATION", "trial")

    # Gradio server settings
    port = args.port or int(os.getenv("GRADIO_PORT", "7860"))

    # Gemini API key from environment (used as default if UI doesn't provide one)
    default_gemini_api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")

    # Auto-generate bearer token if ARM params are provided
    if not bearer_token and arm_subscription_id and arm_resource_group and arm_account_name:
        logging.info("Auto-generating ARM bearer token...")
        bearer_token = generate_arm_token(
            arm_subscription_id,
            arm_resource_group,
            arm_account_name,
        )
        logging.info("Successfully generated ARM bearer token.")

    # Validate authentication configuration
    if not bearer_token and not subscription_key:
        raise ValueError(
            "Authentication missing. Set environment variables in .env file:\n\n"
            "=== Option 1: Trial Account (Free) ===\n"
            "  - AZURE_VIDEO_INDEXER_SUBSCRIPTION_KEY=<your-api-key>\n"
            "  - AZURE_VIDEO_INDEXER_ACCOUNT_ID=<your-account-id>\n"
            "  - AZURE_VIDEO_INDEXER_LOCATION=trial\n"
            "  Get API key from: https://api-portal.videoindexer.ai\n\n"
            "=== Option 2: ARM Account (Paid) ===\n"
            "  - ARM_SUBSCRIPTION_ID=<your-subscription-id>\n"
            "  - ARM_RESOURCE_GROUP=<your-resource-group>\n"
            "  - ARM_ACCOUNT_NAME=<your-account-name>\n"
            "  Plus ONE of:\n"
            "    a) AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID (Service Principal)\n"
            "    b) USE_INTERACTIVE_AUTH=true (Browser login)\n"
            "    c) USE_DEVICE_CODE_AUTH=true (Device code)\n\n"
            "=== Option 3: Manual Bearer Token ===\n"
            "  - AZURE_VIDEO_INDEXER_BEARER_TOKEN=<your-token>\n"
        )

    # Token refresh callback for automatic token renewal on 401 errors
    def refresh_bearer_token():
        """Callback to refresh bearer token when it expires."""
        if arm_subscription_id and arm_resource_group and arm_account_name:
            logging.info("Refreshing bearer token via ARM...")
            new_token = generate_arm_token(
                arm_subscription_id,
                arm_resource_group,
                arm_account_name,
            )
            logging.info("Bearer token refreshed successfully")
            return new_token
        else:
            logging.warning("Cannot refresh token: ARM parameters not configured")
            return None

    # Initialize Azure Video Indexer processor
    logging.info("Initializing Azure Video Indexer Processor...")
    audio_clipper = AzureVideoIndexerProcessor(
        subscription_key=subscription_key or "",
        location=location,
        account_id=account_id,
        bearer_token=bearer_token,
        token_refresh_callback=refresh_bearer_token if bearer_token else None,
    )

    server_name = "127.0.0.1"
    if args.listen:
        server_name = "0.0.0.0"

    def audio_recog(audio_input, sd_switch):
        return audio_clipper.recog(audio_input, sd_switch, None, "", output_dir=None)

    def video_recog(video_input, sd_switch):
        return audio_clipper.video_recog(video_input, sd_switch, "", output_dir=None)

    def mix_recog(video_input, audio_input):
        import shutil
        import tempfile
        import time
        from uuid import uuid4

        audio_state, video_state = None, None
        if video_input is not None:
            # Gradio on Windows may lock uploaded video files (ffmpeg conversion/preview),
            # which can trigger PermissionError when the UI tries to stream the input.
            # Work around by copying the uploaded file to a safe temp path before processing.
            src_path = video_input
            if isinstance(video_input, dict):
                src_path = video_input.get("path") or video_input.get("name")  # best-effort
            if not isinstance(src_path, str) or not src_path:
                raise ValueError(f"Invalid video input: {video_input}")

            suffix = os.path.splitext(src_path)[1] or ".mp4"
            dst_path = os.path.join(tempfile.gettempdir(), f"autoclipper_{uuid4().hex}{suffix}")

            last_err = None
            for _ in range(10):
                try:
                    shutil.copy2(src_path, dst_path)
                    last_err = None
                    break
                except PermissionError as e:
                    last_err = e
                    time.sleep(0.2)
            if last_err:
                logging.warning("Failed to copy uploaded video, using original path: %s", last_err)
                dst_path = src_path

            res_text, res_srt, video_state = video_recog(dst_path, "No")
            return res_srt, video_state, None
        if audio_input is not None:
            res_text, res_srt, audio_state = audio_recog(audio_input, "No")
            return res_srt, None, audio_state

    def llm_inference(system_content, user_content, srt_text, model, apikey):
        SUPPORT_LLM_PREFIX = ["gemini"]
        # Use UI-provided key if available, otherwise fall back to env default
        effective_apikey = apikey.strip() if apikey else ""
        if not effective_apikey:
            effective_apikey = default_gemini_api_key
            if effective_apikey:
                logging.info("Using default Gemini API key from GOOGLE_AI_STUDIO_API_KEY env var")
        if not effective_apikey:
            return "Error: No API key provided. Please enter a Gemini API key or set GOOGLE_AI_STUDIO_API_KEY in .env"

        if model.startswith("gemini"):
            return gemini_call(
                apikey=effective_apikey,
                model=model,
                user_content=user_content + "\n" + srt_text,
                system_content=system_content,
            )
        else:
            logging.error(
                f"LLM name error, only {SUPPORT_LLM_PREFIX} are supported as LLM name prefix."
            )
            return "Error: Unsupported model. Please use Gemini models."

    def AI_clip_segments(
        LLM_res,
        video_state,
        audio_state,
        video_input,
        srt_text,
    ):
        """
        Clip video/audio into multiple story segments based on LLM result.
        Returns a list of video paths and a summary message.
        """
        logging.info("=" * 60)
        logging.info("AI_clip_segments called")
        logging.info(f"video_state: {video_state is not None}")
        logging.info(f"audio_state: {audio_state is not None}")
        logging.info(f"video_input: {video_input}")
        logging.info(f"srt_text length: {len(srt_text) if srt_text else 0}")
        logging.info("=" * 60)

        # Check if we have valid input
        has_state = video_state is not None or audio_state is not None
        has_manual_input = video_input is not None and srt_text and srt_text.strip()

        # Use global MAX_SEGMENTS (defined at module level from env var)

        if not has_state and not has_manual_input:
            logging.error("No valid input: neither ASR state nor manual input")
            results = [None] * MAX_SEGMENTS + [""] * MAX_SEGMENTS
            results.append(
                "Error: Please either run ASR, or upload a video AND paste SRT manually."
            )
            return results

        # Extract timestamps with associated text from LLM result
        llm_segments = extract_timestamps_with_text(LLM_res)
        timestamp_list = [[seg["start_ms"], seg["end_ms"]] for seg in llm_segments]

        logging.info(f"Extracted {len(timestamp_list)} timestamps from LLM result")
        for i, seg in enumerate(llm_segments):
            logging.info(
                f"  Timestamp {i + 1}: {seg['start_ms'] / 1000:.2f}s - {seg['end_ms'] / 1000:.2f}s"
            )
            logging.info(
                f"    Text preview: {seg['text'][:100]}..."
                if len(seg["text"]) > 100
                else f"    Text: {seg['text']}"
            )

        if not timestamp_list:
            logging.error("No timestamps found in LLM result")
            results = [None] * MAX_SEGMENTS + [""] * MAX_SEGMENTS
            results.append("Error: No timestamps found in LLM result.")
            return results

        # Output directory is not customizable, use default (None)
        output_dir = None

        segments = []
        is_video = False

        try:
            if video_state is not None:
                logging.info("Mode: Using ASR video state")
                is_video = True
                segments = audio_clipper.video_clip_segments(
                    state=video_state,
                    output_dir=output_dir,
                    timestamp_list=timestamp_list,
                )
            elif audio_state is not None:
                logging.info("Mode: Using ASR audio state")
                is_video = False
                segments = audio_clipper.audio_clip_segments(
                    state=audio_state,
                    output_dir=output_dir,
                    timestamp_list=timestamp_list,
                )
            elif video_input is not None:
                logging.info("Mode: Manual mode with video file")
                is_video = True
                manual_state = {
                    "video_filename": video_input,
                    "sentences": [],
                }
                segments = audio_clipper.video_clip_segments(
                    state=manual_state,
                    output_dir=output_dir,
                    timestamp_list=timestamp_list,
                )
        except Exception as e:
            logging.error(f"Error clipping segments: {e}", exc_info=True)
            results = [None] * MAX_SEGMENTS + [""] * MAX_SEGMENTS
            results.append(f"Error: {str(e)}")
            return results

        logging.info(f"Total segments clipped: {len(segments)}, is_video: {is_video}")

        # Build video paths and transcripts lists
        video_paths = []
        transcripts = []
        for i, seg in enumerate(segments):
            video_path = seg.get("video_path")
            exists = os.path.exists(video_path) if video_path else False
            file_size = os.path.getsize(video_path) if exists else 0  # type: ignore

            # Get transcript from LLM result (more reliable than ASR-based transcript)
            llm_text = ""
            if i < len(llm_segments):
                llm_text = llm_segments[i].get("text", "")

            # Fallback to segment's own transcript if LLM text is empty
            srt_content = seg.get("srt", "")
            transcript = seg.get("transcript", "")

            logging.info(f"Segment {i + 1}:")
            logging.info(f"  video_path: {video_path}")
            logging.info(f"  exists: {exists}")
            logging.info(f"  file_size: {file_size} bytes")
            logging.info(f"  time range: {seg['start']:.2f}s - {seg['end']:.2f}s")
            logging.info(f"  llm_text length: {len(llm_text)}")
            logging.info(f"  transcript length: {len(transcript)}")

            if is_video and exists:
                video_paths.append(video_path)
                # Priority: LLM text > SRT > transcript
                subtitle_text = (
                    llm_text if llm_text else (srt_content if srt_content else transcript)
                )
                transcripts.append(subtitle_text)

        # Build summary message
        if segments:
            message = f"✅ Successfully clipped {len(segments)} story(s).\n"
            for i, seg in enumerate(segments):
                video_path = seg.get("video_path", "N/A")
                exists = os.path.exists(video_path) if video_path else False
                status = "✓" if exists else "✗"
                message += f"\n{status} Story {i + 1}: {seg['start']:.2f}s - {seg['end']:.2f}s"
                if exists:
                    message += f"\n   Path: {video_path}"
        else:
            message = "⚠️ No stories could be clipped."

        logging.info(f"Returning {len(video_paths)} video paths")
        logging.info("=" * 60)

        # Return: videos (MAX_SEGMENTS) + transcripts (MAX_SEGMENTS) + message (1)
        results = []
        # Add video paths
        for i in range(MAX_SEGMENTS):
            if i < len(video_paths):
                results.append(video_paths[i])
            else:
                results.append(None)
        # Add transcripts
        for i in range(MAX_SEGMENTS):
            if i < len(transcripts):
                results.append(transcripts[i])
            else:
                results.append("")
        results.append(message)
        return results

    # gradio interface
    theme = gr.Theme.load("funclip/utils/theme.json")

    with gr.Blocks() as funclip_service:
        # Custom CSS for styling
        gr.HTML("""
        <style>
            /* Hide the entire row when the video inside is empty */
            .video-container:has(.empty[aria-label="Empty value"]) {
                display: none !important;
            }

            /* Make video players taller */
            .video-container video {
                min-height: 400px !important;
                height: 400px !important;
            }

            /* Yellow theme for video container labels */
            .video-container .label-wrap span {
                background-color: #f5c542 !important;
                color: #1a1a1a !important;
                font-weight: bold !important;
            }

            /* Yellow border for video players */
            .video-container .video-container-inner,
            .video-container .wrap {
                border: 2px solid #f5c542 !important;
                border-radius: 8px !important;
            }
        </style>
        """)
        gr.Markdown(top_md_1)
        # gr.Markdown(top_md_2)
        gr.Markdown(top_md_3)
        gr.Markdown(top_md_4)
        video_state, audio_state = gr.State(), gr.State()
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    # Use File input instead of Video to avoid Windows file-lock issues
                    # when Gradio converts/previews uploaded videos (PermissionError on output.mp4).
                    video_input = gr.File(
                        label="Video Input", file_types=["video"], type="filepath"
                    )
                    audio_input = gr.Audio(label="Audio Input")
                with gr.Column():
                    recog_button = gr.Button("1️⃣ ASR (costs money)", variant="stop")
                video_srt_output = gr.Textbox(
                    label="SRT Subtitles (can paste manually to skip ASR)",
                    lines=10,
                    interactive=True,
                    placeholder="You can paste SRT content here manually to skip ASR step...",
                )
            with gr.Column():
                with gr.Tab("LLM Clipping"):
                    with gr.Column():
                        prompt_head = gr.Textbox(
                            label="Prompt System (modify as needed, try not to change the main body and requirements)",
                            value=(
                                """
                            Role: You are a professional News Editor and Video Clipper.
                            Objective: Extract ONLY the hard news segments from the SRT subtitles.
                            Critical Exclusion Rules (What to DELETE):
                            Ignore Ads: Delete all commercial advertisements and sponsorships.
                            Ignore Routine Updates: DELETE all Traffic reports, Weather forecasts, and Stock market/Currency snapshots.
                            Ignore Intro/Outro: Delete station intros or generic show openers unless they lead immediately into a news story.
                            Segmentation Logic:
                            Treat each distinct news story as a separate segment.
                            Cut when the topic changes (e.g., from a crime story to a political story).
                            Merge consecutive lines within the same story.
                            Output Format:
                            Strictly follow this format:
                            [start time–end time] Text content of the news story
                            Note that the connector between the start time and end time must be “–”.hat the connector between the start time and end time must be “–”."""
                            ),
                        )
                        prompt_head2 = gr.Textbox(
                            label="Prompt User (no need to modify, will automatically concatenate SRT subtitles from bottom left)",
                            value=("This is the video SRT subtitle to be clipped:"),
                        )
                        with gr.Column():
                            with gr.Row():
                                llm_model = gr.Dropdown(
                                    choices=[
                                        "gemini-3-pro-preview",
                                        "gemini-3-flash-preview",
                                    ],
                                    value="gemini-3-flash-preview",
                                    label="LLM Model Name",
                                    allow_custom_value=True,
                                )
                                apikey_input = gr.Textbox(
                                    label="API Key",
                                    placeholder="Use system default API key if you don't have one",
                                    type="password",
                                )
                            llm_button = gr.Button(
                                "2️⃣ LLM Inference ",
                                variant="stop",
                            )
                        llm_result = gr.Textbox(label="LLM Clipper Result", lines=20)
                        llm_clip_button = gr.Button("3️⃣ 🎬 AI Clip", variant="stop")

                # Story segments output
                gr.Markdown("### 📖 Clipped Stories")
                clip_segments_message = gr.Textbox(
                    label="Clipping Summary", lines=12, interactive=False
                )

                # Create video players and transcript textboxes (hidden by CSS when empty)
                # Uses global MAX_SEGMENTS (defined at module level from env var)
                segment_videos = []
                segment_transcripts = []
                for i in range(MAX_SEGMENTS):
                    with gr.Row(elem_classes=["video-container"]):
                        with gr.Column(scale=2):
                            vid = gr.Video(
                                label=f"Story {i + 1}",
                                interactive=False,
                            )
                            segment_videos.append(vid)
                        with gr.Column(scale=1):
                            txt = gr.Textbox(
                                label=f"Transcript {i + 1}",
                                lines=6,
                                interactive=False,
                            )
                            segment_transcripts.append(txt)

        recog_button.click(
            mix_recog,
            inputs=[
                video_input,
                audio_input,
            ],
            outputs=[video_srt_output, video_state, audio_state],
        )
        llm_button.click(
            llm_inference,
            inputs=[prompt_head, prompt_head2, video_srt_output, llm_model, apikey_input],
            outputs=[llm_result],
        )

        # Helper function to clear all segment videos and transcripts before re-clipping
        # This forces Gradio to reload files instead of using cached versions
        def clear_segments():
            """Clear all video players and transcripts to force reload on next clip."""
            return [None] * MAX_SEGMENTS + [""] * MAX_SEGMENTS + ["⏳ Clipping in progress..."]

        # AI Clip button - first clear all segments, then run clipping
        # This two-step approach ensures video players don't show stale cached content
        llm_clip_button.click(
            clear_segments,
            inputs=[],
            outputs=segment_videos + segment_transcripts + [clip_segments_message],
        ).then(
            AI_clip_segments,
            inputs=[
                llm_result,
                video_state,
                audio_state,
                video_input,
                video_srt_output,
            ],
            outputs=segment_videos + segment_transcripts + [clip_segments_message],
        )

    # Start gradio service
    logging.info(f"Starting Gradio server on {server_name}:{port}")

    # Set up authentication if USERNAME and PASSWORD are configured
    auth = None
    if APP_USERNAME and APP_PASSWORD:
        logging.info(f"Authentication enabled (username: {APP_USERNAME})")

        def check_credentials(username, password):
            return username == APP_USERNAME and password == APP_PASSWORD

        auth = check_credentials
    else:
        logging.info("No authentication (USERNAME or PASSWORD env var not set)")

    if args.listen:
        funclip_service.launch(
            share=args.share,
            server_port=port,
            server_name=server_name,
            inbrowser=False,
            theme=theme,
            auth=auth,
        )
    else:
        funclip_service.launch(
            share=args.share,
            server_port=port,
            server_name=server_name,
            theme=theme,
            auth=auth,
        )
