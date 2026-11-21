"""
YouTube API & Transcript Extraction Module (`youtube_api.py`).
Handles:
- Normalizing channel URLs, @handles, playlists, and video URLs (`_normalize_channel_url`)
- Discovering channel videos via `yt-dlp` (`get_channel_videos`, `get_channel_video_ids`)
- On-demand channel-scoped video search (`search_channel_videos_for_query`)
- 3-Tier transcript & course syllabus extraction (`get_transcript`, `_extract_transcript_via_ytdlp`)
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

from youtube_rag.chunking import extract_content_keywords


def _normalize_channel_url(channel_url: str) -> str:
    """
    Normalize any user input into a valid YouTube URL:
    - '@freecodecamp' or 'freecodecamp' -> 'https://www.youtube.com/@freecodecamp/videos'
    - 'https://www.youtube.com/@freecodecamp' -> 'https://www.youtube.com/@freecodecamp/videos'
    - Single video URLs ('watch?v=' or 'youtu.be/') and playlists ('playlist?list=') are preserved as-is.
    """
    url = channel_url.strip()
    if not url:
        return url

    if not url.startswith(("http://", "https://")):
        if url.startswith(("www.youtube.com", "youtube.com", "youtu.be")):
            url = "https://" + url
        else:
            handle = url.lstrip("@").strip("/")
            return f"https://www.youtube.com/@{handle}/videos"

    if "watch?v=" in url or "youtu.be/" in url or "playlist?list=" in url or "/shorts/" in url or "/search?" in url:
        return url

    if any(p in url for p in ("youtube.com/@", "youtube.com/c/", "youtube.com/channel/", "youtube.com/user/")):
        if not any(url.rstrip("/").endswith(suffix) for suffix in ("/videos", "/streams", "/shorts", "/playlists")):
            url = url.rstrip("/") + "/videos"
    return url


def _channel_base_url(channel_url: str) -> str:
    """Extract the base channel URL (e.g. 'https://www.youtube.com/@freecodecamp') for channel-scoped search."""
    norm = _normalize_channel_url(channel_url)
    for suffix in ("/videos", "/streams", "/shorts", "/playlists"):
        if norm.endswith(suffix):
            return norm[: -len(suffix)]
    return norm.rstrip("/")


def get_channel_videos(channel_url: str, max_videos: int = 100) -> list[dict[str, str]]:
    """
    Fetch video metadata (id, title, channel, url) from ANY YouTube channel, search URL, playlist, or video URL using yt-dlp.
    """
    normalized_url = _normalize_channel_url(channel_url)
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "socket_timeout": 6,
        "retries": 1,
        "extractor_retries": 1,
        "playlistend": max_videos,
    }
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(normalized_url, download=False)
        if not info:
            return []

        channel_name = (
            info.get("channel")
            or info.get("uploader")
            or info.get("title")
            or normalized_url
        )

        if "entries" in info and info["entries"]:
            videos = []
            for entry in info["entries"]:
                if not entry:
                    continue
                if "entries" in entry and entry["entries"]:
                    for sub in entry["entries"]:
                        if not sub:
                            continue
                        vid = sub.get("id")
                        if vid and len(vid) == 11:
                            videos.append(
                                {
                                    "id": vid,
                                    "title": sub.get("title") or f"Video {vid}",
                                    "channel": sub.get("channel") or sub.get("uploader") or channel_name,
                                    "description": sub.get("description") or "",
                                    "url": f"https://youtube.com/watch?v={vid}",
                                }
                            )
                    continue

                vid = entry.get("id")
                if vid and len(vid) == 11:
                    videos.append(
                        {
                            "id": vid,
                            "title": entry.get("title") or f"Video {vid}",
                            "channel": entry.get("channel") or entry.get("uploader") or channel_name,
                            "description": entry.get("description") or "",
                            "url": f"https://youtube.com/watch?v={vid}",
                        }
                    )
            return videos[:max_videos]

        vid = info.get("id")
        if vid:
            return [
                {
                    "id": vid,
                    "title": info.get("title") or f"Video {vid}",
                    "channel": channel_name,
                    "description": info.get("description") or "",
                    "url": f"https://youtube.com/watch?v={vid}",
                }
            ]
        return []


def search_channel_videos_for_query(
    channel_url: str,
    question: str,
    max_search_results: int = 3,
) -> list[dict[str, str]]:
    """
    Search INSIDE a specific YouTube channel for videos matching the user's question.
    Uses YouTube's channel search endpoint (`/@channel/search?query=...`) via yt-dlp to locate
    the exact videos on that channel that teach the requested topic.
    """
    keywords = extract_content_keywords(question)
    if not keywords:
        return []
    query_str = " ".join(keywords[:6])
    base_url = _channel_base_url(channel_url)

    if any(p in base_url for p in ("youtube.com/@", "youtube.com/c/", "youtube.com/channel/", "youtube.com/user/")):
        encoded_q = urllib.parse.quote_plus(query_str)
        search_tab_url = f"{base_url}/search?query={encoded_q}"
        try:
            vids = get_channel_videos(search_tab_url, max_videos=max_search_results)
            if vids:
                return vids
        except Exception:
            pass

    handle_match = re.search(r"@([A-Za-z0-9_.-]+)", base_url)
    channel_hint = handle_match.group(1) if handle_match else base_url
    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "socket_timeout": 6,
        "retries": 1,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_search_results}:{channel_hint} {query_str}", download=False)
            entries = (info or {}).get("entries") or []
            results = []
            for entry in entries:
                if not entry:
                    continue
                vid = entry.get("id")
                if vid and len(vid) == 11:
                    results.append(
                        {
                            "id": vid,
                            "title": entry.get("title") or f"Video {vid}",
                            "channel": entry.get("channel") or entry.get("uploader") or channel_hint,
                            "description": entry.get("description") or "",
                            "url": f"https://youtube.com/watch?v={vid}",
                        }
                    )
            return results
    except Exception:
        return []


def get_channel_video_ids(channel_url: str, max_videos: int = 100) -> list[str]:
    """Fetch all video IDs from a YouTube channel (exact tutorial signature)."""
    return [v["id"] for v in get_channel_videos(channel_url, max_videos=max_videos)]


def _extract_transcript_via_ytdlp(video_id: str) -> Optional[str]:
    """
    Fallback transcript & metadata extractor using yt-dlp when youtube-transcript-api
    encounters missing standard captions or language-code mismatches.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 6,
        "retries": 1,
        "extractor_retries": 1,
        "writesubtitles": True,
        "writeautomaticsub": True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return None

            for sub_field in ("subtitles", "automatic_captions"):
                subs = info.get(sub_field) or {}
                lang_keys = [k for k in subs.keys() if k.startswith("en")] + list(subs.keys())
                for lang in lang_keys:
                    formats = subs.get(lang) or []
                    json3_fmt = next((f for f in formats if f.get("ext") == "json3" and f.get("url")), None)
                    if json3_fmt:
                        req = urllib.request.Request(
                            json3_fmt["url"],
                            headers={"User-Agent": "Mozilla/5.0"},
                        )
                        with urllib.request.urlopen(req, timeout=10) as resp:
                            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                        events = data.get("events", [])
                        words = []
                        for ev in events:
                            for seg in ev.get("segs", []):
                                utf8 = seg.get("utf8", "").strip()
                                if utf8 and utf8 != "\n":
                                    words.append(utf8)
                        text = " ".join(words).strip()
                        if len(text) > 50:
                            return text

            parts = []
            if info.get("title"):
                parts.append(f"Video Title: {info['title']}.")
            if info.get("chapters"):
                chapter_titles = [
                    f"{ch.get('title', '')}" for ch in info["chapters"] if ch.get("title")
                ]
                if chapter_titles:
                    parts.append("Course Chapters & Topics Covered: " + "; ".join(chapter_titles) + ".")
            if info.get("description"):
                parts.append("Video Description & Syllabus: " + info["description"])

            combined = "\n".join(parts).strip()
            return combined if len(combined) > 40 else None
    except Exception:
        return None


def get_transcript(video_id: str) -> Optional[str]:
    """
    Fetch transcript for a single video using a multi-tier strategy so ANY channel works:
    Tier 1: youtube-transcript-api (checks English variants first, then any available transcript track)
    Tier 2: yt-dlp subtitle & rich course syllabus/chapter fallback
    """
    preferred_langs = ["en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU"]

    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_langs)
            text = " ".join([chunk["text"] for chunk in transcript]).strip()
            if text:
                return text
        else:
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=preferred_langs)
            snippets = [s.text if hasattr(s, "text") else s["text"] for s in fetched]
            text = " ".join(snippets).strip()
            if text:
                return text
    except Exception:
        pass

    try:
        if hasattr(YouTubeTranscriptApi, "list_transcripts"):
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for tr in transcript_list:
                fetched = tr.fetch()
                text = " ".join([c["text"] if isinstance(c, dict) else c.text for c in fetched]).strip()
                if text:
                    return text
        else:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            for tr in transcript_list:
                fetched = tr.fetch()
                snippets = [s.text if hasattr(s, "text") else s["text"] for s in fetched]
                text = " ".join(snippets).strip()
                if text:
                    return text
    except Exception:
        pass

    return _extract_transcript_via_ytdlp(video_id)
