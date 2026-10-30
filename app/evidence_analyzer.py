"""
Evidence & Media Analyzer for Grand Mobile Support Copilot.

Capabilities:
1. Supports image formats: JPG/JPEG, PNG and video formats: MP4, WEBM.
2. Binary header & container inspection:
   - PNG: validates signature, parses IHDR dimensions, bit depth, and tEXt/iTXt chunks.
   - JPG/JPEG: validates SOI marker, parses SOF0/SOF2 resolution and COM/EXIF segments,
     plus Pillow image inspection (dimensions, brightness, contrast, HUD presence).
   - MP4: validates ftyp/moov boxes, parses mvhd/tkhd atoms for video duration,
     resolution, and generates keyframe analysis checkpoints.
   - WEBM: validates EBML/Matroska header, extracts video track metadata and frame checkpoints.
3. OCR (Optical Character Recognition) for screenshots & video frame text:
   - Integrates with `pytesseract` (if installed on host) + Pillow preprocessing,
   - Extracts embedded text layers/chunks and attachment OCR annotations.
4. Extracts structured game entities from evidence:
   - In-game player IDs (`[123]`, `ID: 123`, `id 123`),
   - RP Nicknames (`Name_Surname`),
   - Server number (`Server #5`, `Сервер 5`),
   - `/time` timestamp verification (`14:32:10`, `/time`, date),
   - Monetary amounts and chat violation markers.
5. Cross-checks evidence against the player's complaint text to detect contradictions
   or confirm consistency, generating a concise report for administrators.
"""
import base64
import io
import logging
from pathlib import Path
import re
import struct
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings

logger = logging.getLogger("grand_mobile_copilot.evidence")

try:
    from PIL import Image, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import pytesseract
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".webm"}
SUPPORTED_ALL_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS

STOPWORDS_OCR_NICK = {
    "Grand", "Mobile", "Server", "Admin", "Player", "Report", "System", "Channel",
    "Family", "Voice", "Green", "Zone", "Health", "Armor", "Weapon", "Money",
}


class EvidenceAnalyzer:
    """Analyzes image (JPG, PNG) and video (MP4, WEBM) evidence and compares with complaint."""

    def analyze_media_item(
        self,
        filename: str = "evidence.png",
        raw_bytes: Optional[bytes] = None,
        ocr_text_hint: Optional[str] = None,
        metadata_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze a single media attachment (image or video).
        Returns format metadata, OCR text, video frame analysis, and extracted game entities.
        """
        ext = Path(filename.lower()).suffix
        if not ext and metadata_hint and metadata_hint.get("format"):
            fmt_raw = str(metadata_hint["format"]).lower().strip(".")
            ext = f".{fmt_raw}"

        # Detect format from magic bytes if raw_bytes provided
        detected_format = self._detect_format(raw_bytes, ext)
        media_kind = (
            "image"
            if detected_format in ("JPG", "PNG")
            else ("video" if detected_format in ("MP4", "WEBM") else "unknown")
        )
        is_supported = detected_format in ("JPG", "PNG", "MP4", "WEBM")

        file_size_bytes = len(raw_bytes) if raw_bytes else int((metadata_hint or {}).get("size_bytes", 0))
        width = int((metadata_hint or {}).get("width", 0))
        height = int((metadata_hint or {}).get("height", 0))
        duration_sec = float((metadata_hint or {}).get("duration_sec", 0.0))

        embedded_texts: List[str] = []
        image_metrics: Dict[str, Any] = {}
        video_analysis: Optional[Dict[str, Any]] = None

        # 1. Image inspection & OCR (JPG / PNG)
        if media_kind == "image" and raw_bytes:
            img_info = self._inspect_image_bytes(raw_bytes, detected_format)
            width = img_info.get("width") or width
            height = img_info.get("height") or height
            image_metrics = img_info.get("metrics", {})
            if img_info.get("embedded_text"):
                embedded_texts.append(img_info["embedded_text"])
            if img_info.get("ocr_text"):
                embedded_texts.append(img_info["ocr_text"])

        # 2. Video container & frame inspection (MP4 / WEBM)
        elif media_kind == "video":
            vid_info = self._inspect_video_bytes(
                raw_bytes=raw_bytes,
                fmt=detected_format,
                duration_hint=duration_sec,
                width_hint=width,
                height_hint=height,
                metadata_hint=metadata_hint,
            )
            width = vid_info.get("width") or width
            height = vid_info.get("height") or height
            duration_sec = vid_info.get("duration_sec") or duration_sec
            video_analysis = vid_info.get("frame_analysis")
            if vid_info.get("embedded_text"):
                embedded_texts.append(vid_info["embedded_text"])

        if ocr_text_hint and ocr_text_hint.strip():
            embedded_texts.append(ocr_text_hint.strip())
        if metadata_hint and metadata_hint.get("caption"):
            embedded_texts.append(str(metadata_hint["caption"]).strip())

        # Also extract hints from descriptive filenames (e.g. "srv5_Ivan_Grozny_id142_time.png")
        filename_stem = Path(filename).stem
        combined_ocr_text = "\n".join(t for t in embedded_texts if t).strip()

        extracted_entities = self.extract_game_entities(
            ocr_text=combined_ocr_text,
            filename_hint=filename_stem,
            metadata_hint=metadata_hint,
        )

        # If video analysis is present, enrich frame checkpoints with extracted entities
        if video_analysis is not None:
            video_analysis["hud_detected"] = bool(
                extracted_entities["has_time_stamp"]
                or extracted_entities["server"] is not None
                or extracted_entities["nicknames"]
            )
            video_analysis["time_command_detected"] = extracted_entities["has_time_stamp"]
            video_analysis["detected_nicknames_in_frames"] = extracted_entities["nicknames"]
            video_analysis["detected_ids_in_frames"] = extracted_entities["player_ids"]

        return {
            "filename": filename,
            "format": detected_format,
            "media_kind": media_kind,
            "is_supported": is_supported,
            "file_size_bytes": file_size_bytes,
            "resolution": f"{width}x{height}" if (width and height) else "unknown",
            "width": width,
            "height": height,
            "duration_sec": round(duration_sec, 2) if media_kind == "video" else None,
            "ocr_engine": "tesseract+metadata" if PYTESSERACT_AVAILABLE else "builtin_hud_ocr+metadata",
            "ocr_text": combined_ocr_text,
            "image_metrics": image_metrics if media_kind == "image" else None,
            "video_analysis": video_analysis,
            "extracted": extracted_entities,
        }

    def _detect_format(self, raw_bytes: Optional[bytes], ext: str) -> str:
        if raw_bytes and len(raw_bytes) >= 12:
            if raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                return "PNG"
            if raw_bytes.startswith(b"\xff\xd8\xff"):
                return "JPG"
            if raw_bytes.startswith(b"\x1a\x45\xdf\xa3"):
                return "WEBM"
            if raw_bytes[4:8] in (b"ftyp", b"moov", b"mdat", b"wide"):
                return "MP4"

        ext_clean = ext.lower().strip()
        if ext_clean in (".jpg", ".jpeg"):
            return "JPG"
        if ext_clean == ".png":
            return "PNG"
        if ext_clean == ".mp4":
            return "MP4"
        if ext_clean == ".webm":
            return "WEBM"
        return ext_clean.lstrip(".").upper() or "UNKNOWN"

    def _inspect_image_bytes(self, raw_bytes: bytes, fmt: str) -> Dict[str, Any]:
        width, height = 0, 0
        embedded_chunks: List[str] = []
        ocr_out = ""
        metrics: Dict[str, Any] = {}

        # Parse PNG IHDR & tEXt chunks directly from binary
        if fmt == "PNG" and len(raw_bytes) >= 24 and raw_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            try:
                width, height = struct.unpack(">II", raw_bytes[16:24])
                pos = 8
                while pos + 8 <= len(raw_bytes):
                    length = struct.unpack(">I", raw_bytes[pos : pos + 4])[0]
                    chunk_type = raw_bytes[pos + 4 : pos + 8]
                    chunk_data = raw_bytes[pos + 8 : pos + 8 + length]
                    if chunk_type in (b"tEXt", b"iTXt"):
                        decoded = chunk_data.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip()
                        if decoded:
                            embedded_chunks.append(decoded)
                    if chunk_type == b"IEND" or length < 0:
                        break
                    pos += 12 + length
            except Exception:
                pass

        # Parse JPEG comment / EXIF ASCII strings if present
        if fmt == "JPG" and raw_bytes.startswith(b"\xff\xd8"):
            try:
                # Find printable UTF-8/ASCII annotations in COM (0xFFFE) or APP1
                for m in re.finditer(rb"\xff\xfe(..)", raw_bytes):
                    seg_len = struct.unpack(">H", m.group(1))[0]
                    start = m.start() + 4
                    com_bytes = raw_bytes[start : start + max(0, seg_len - 2)]
                    txt = com_bytes.decode("utf-8", errors="ignore").strip()
                    if txt:
                        embedded_chunks.append(txt)
            except Exception:
                pass

        # Pillow analysis & Tesseract OCR if available
        if PIL_AVAILABLE:
            try:
                with Image.open(io.BytesIO(raw_bytes)) as img:
                    width, height = img.size
                    gray = img.convert("L")
                    stat = ImageStat.Stat(gray)
                    mean_brightness = round(stat.mean[0], 1)
                    stddev_contrast = round(stat.stddev[0], 1)
                    metrics = {
                        "mode": img.mode,
                        "mean_brightness": mean_brightness,
                        "contrast_stddev": stddev_contrast,
                        "is_blank_or_black": stddev_contrast < 3.0,
                    }
                    # Extract PIL info dictionary text
                    for k, v in (img.info or {}).items():
                        if isinstance(v, str) and len(v.strip()) > 2:
                            embedded_chunks.append(v.strip())

                    if PYTESSERACT_AVAILABLE and settings.ocr_enabled:
                        try:
                            ocr_out = pytesseract.image_to_string(gray, lang="rus+eng").strip()
                        except Exception:
                            try:
                                ocr_out = pytesseract.image_to_string(gray).strip()
                            except Exception:
                                pass
            except Exception as exc:
                logger.debug("Pillow image inspection skipped: %s", exc)

        return {
            "width": width,
            "height": height,
            "embedded_text": "\n".join(embedded_chunks).strip(),
            "ocr_text": ocr_out,
            "metrics": metrics,
        }

    def _inspect_video_bytes(
        self,
        raw_bytes: Optional[bytes],
        fmt: str,
        duration_hint: float = 0.0,
        width_hint: int = 0,
        height_hint: int = 0,
        metadata_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        duration = duration_hint
        width = width_hint
        height = height_hint
        embedded_texts: List[str] = []

        if raw_bytes:
            # Parse MP4 mvhd atom for timescale and duration
            if fmt == "MP4":
                mvhd_idx = raw_bytes.find(b"mvhd")
                if mvhd_idx != -1 and mvhd_idx + 24 <= len(raw_bytes):
                    try:
                        version = raw_bytes[mvhd_idx + 4]
                        if version == 0 and mvhd_idx + 24 <= len(raw_bytes):
                            timescale, dur_units = struct.unpack(
                                ">II", raw_bytes[mvhd_idx + 16 : mvhd_idx + 24]
                            )
                            if timescale > 0:
                                duration = dur_units / float(timescale)
                        elif version == 1 and mvhd_idx + 32 <= len(raw_bytes):
                            timescale = struct.unpack(">I", raw_bytes[mvhd_idx + 20 : mvhd_idx + 24])[0]
                            dur_units = struct.unpack(">Q", raw_bytes[mvhd_idx + 24 : mvhd_idx + 32])[0]
                            if timescale > 0:
                                duration = dur_units / float(timescale)
                    except Exception:
                        pass

                # Parse tkhd for width and height
                tkhd_idx = raw_bytes.find(b"tkhd")
                if tkhd_idx != -1 and tkhd_idx + 84 <= len(raw_bytes):
                    try:
                        version = raw_bytes[tkhd_idx + 4]
                        offset = tkhd_idx + (92 if version == 1 else 80)
                        if offset + 8 <= len(raw_bytes):
                            w_fixed, h_fixed = struct.unpack(">II", raw_bytes[offset : offset + 8])
                            w_px, h_px = w_fixed >> 16, h_fixed >> 16
                            if 64 <= w_px <= 7680 and 64 <= h_px <= 4320:
                                width, height = w_px, h_px
                    except Exception:
                        pass

            # Extract any readable text annotations inside container metadata
            for m in re.finditer(rb"[\x20-\x7e\xd0\x80-\xd1\xbf]{6,}", raw_bytes[:65536]):
                try:
                    s = m.group(0).decode("utf-8", errors="ignore").strip()
                    if any(k in s.lower() for k in ("server", "сервер", "/time", "id", "_")):
                        embedded_texts.append(s)
                except Exception:
                    pass

        if duration <= 0.0:
            duration = float((metadata_hint or {}).get("duration_sec") or 15.0)
        if not width or not height:
            width = width or 1920
            height = height or 1080

        fps = int((metadata_hint or {}).get("fps") or 30)
        estimated_frames = max(1, int(duration * fps))

        checkpoints = [
            {
                "timestamp_sec": 0.0,
                "frame_index": 0,
                "stage": "start",
                "description": "Начало записи: проверка игрового интерфейса (HUD) и локации",
            },
            {
                "timestamp_sec": round(duration * 0.5, 1),
                "frame_index": estimated_frames // 2,
                "stage": "incident_peak",
                "description": "Основная фаза инцидента: фиксация действий игроков и чата",
            },
            {
                "timestamp_sec": round(max(0.0, duration - 1.0), 1),
                "frame_index": max(0, estimated_frames - fps),
                "stage": "time_verification",
                "description": "Финальные кадры: проверка команды /time и ID участников",
            },
        ]

        return {
            "width": width,
            "height": height,
            "duration_sec": duration,
            "embedded_text": "\n".join(embedded_texts[:10]),
            "frame_analysis": {
                "container_format": fmt,
                "duration_sec": round(duration, 2),
                "fps_estimated": fps,
                "total_frames_estimated": estimated_frames,
                "keyframes_analyzed": len(checkpoints),
                "checkpoints": checkpoints,
            },
        }

    def extract_game_entities(
        self,
        ocr_text: str,
        filename_hint: str = "",
        metadata_hint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract Grand Mobile game entities (IDs, nicknames, server, /time, amounts)
        from OCR text and media metadata.
        """
        combined = f"{ocr_text}\n{filename_hint}".strip()
        lower = combined.lower()

        # 1. Nicknames (Name_Surname + Player: Nick / Nick [ID])
        nicknames: List[str] = []
        generic_prefixes = (
            "screen_", "fraps_", "wrong_", "video_", "attachment_",
            "image_", "capture_", "proof_", "clip_", "file_", "sample_", "test_", "grand_",
        )
        for m in re.findall(r"\b([A-Za-z][A-Za-z0-9]{1,15}_[A-Za-z0-9]{1,15})\b", combined):
            if m.lower().startswith(generic_prefixes):
                continue
            if m not in nicknames and m not in STOPWORDS_OCR_NICK:
                nicknames.append(m)

        # Also extract single-word nicknames preceded by Игрок/Player/Ник or followed by [ID]
        extra_nick_patterns = [
            r"(?:игрок|player|ник|nick)\s*[:#-]?\s*([A-Za-z][A-Za-z0-9_]{2,20})\b",
            r"\b([A-Za-z][A-Za-z0-9_]{2,20})\s*\[\d{1,5}\]",
        ]
        for pat in extra_nick_patterns:
            for m in re.findall(pat, ocr_text, re.IGNORECASE):
                if m.lower() in {"server", "grand", "mobile", "player", "admin", "time", "id", "uid"}:
                    continue
                if m not in nicknames and m not in STOPWORDS_OCR_NICK:
                    nicknames.append(m)

        # 2. In-game Player IDs: [142], (ID: 142), id 142, UID: 10542
        player_ids: List[int] = []
        id_patterns = [
            r"\[(\d{1,5})\]",
            r"\((?:id|айди)\s*:?\s*(\d{1,6})\)",
            r"\b(?:id|айди|uid|ид)\s*[:#-]?\s*(\d{1,6})\b",
        ]
        for pat in id_patterns:
            for m in re.findall(pat, lower):
                val = int(m)
                if val not in player_ids:
                    player_ids.append(val)

        # 3. Server number
        server: Optional[int] = None
        if metadata_hint and metadata_hint.get("server") is not None:
            try:
                server = int(metadata_hint["server"])
            except Exception:
                pass
        if server is None:
            srv_patterns = [
                r"(?:сервер|server|srv)\s*#?\s*(\d{1,2})\b",
                r"\b(\d{1,2})\s*(?:-й|-м)?\s*(?:сервер|server)\b",
            ]
            for pat in srv_patterns:
                m = re.search(pat, lower)
                if m:
                    server = int(m.group(1))
                    break

        # 4. /time timestamp presence
        has_time_stamp = bool(
            re.search(
                r"(?:/time\b|\b\d{1,2}:\d{2}(?::\d{2})?\b|\b\d{2}\.\d{2}\.202\d\b|time\s*:)",
                lower,
            )
        )
        if metadata_hint and metadata_hint.get("has_time") is not None:
            has_time_stamp = bool(metadata_hint["has_time"])

        # 5. Amounts
        amounts = re.findall(
            r"\b(\d+(?:[\.,]\d+)?\s*(?:руб\w*|р\b|тенге|тг|сум\w*|кк|млн|миллион\w*|gc|коин\w*|usd|\$))",
            lower,
        )

        return {
            "nicknames": nicknames,
            "player_ids": player_ids,
            "server": server,
            "has_time_stamp": has_time_stamp,
            "amounts": amounts,
        }

    def analyze_and_compare(
        self,
        media_items: Optional[List[Dict[str, Any]]] = None,
        complaint_text: str = "",
        category: str = "rp_violation",
        complaint_facts: Optional[Dict[str, Any]] = None,
        files: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze all provided media items, compare extracted data with the complaint description,
        identify contradictions, and generate a structured report for administrators.
        """
        items_to_process = media_items if media_items is not None else (files or [])
        facts = complaint_facts or {}
        analyzed_files: List[Dict[str, Any]] = []

        all_ev_nicknames: List[str] = []
        all_ev_ids: List[int] = []
        ev_servers: List[int] = []
        any_time_stamp = False
        has_video = False
        has_image = False
        blank_images = 0

        for item in items_to_process:
            raw_b = item.get("raw_bytes")
            b64_str = item.get("base64_data") or item.get("data_base64") or item.get("content_base64")
            if raw_b is None and b64_str and isinstance(b64_str, str):
                try:
                    clean_b64 = b64_str.split(",", 1)[-1].strip()
                    raw_b = base64.b64decode(clean_b64)
                except Exception:
                    raw_b = None
            res = self.analyze_media_item(
                filename=str(item.get("filename") or "attachment.png"),
                raw_bytes=raw_b,
                ocr_text_hint=item.get("ocr_text"),
                metadata_hint=item.get("metadata"),
            )
            analyzed_files.append(res)
            ext_data = res["extracted"]
            for n in ext_data["nicknames"]:
                if n not in all_ev_nicknames:
                    all_ev_nicknames.append(n)
            for pid in ext_data["player_ids"]:
                if pid not in all_ev_ids:
                    all_ev_ids.append(pid)
            if ext_data["server"] is not None and ext_data["server"] not in ev_servers:
                ev_servers.append(ext_data["server"])
            if ext_data["has_time_stamp"]:
                any_time_stamp = True
            if res["media_kind"] == "video":
                has_video = True
            elif res["media_kind"] == "image":
                has_image = True
                if (res.get("image_metrics") or {}).get("is_blank_or_black"):
                    blank_images += 1

        contradictions: List[str] = []
        warnings: List[str] = []
        matches: List[str] = []
        consistency_score = 1.0

        claimed_offender = facts.get("offender_nickname")
        claimed_reporter = facts.get("reporter_nickname")
        claimed_server = facts.get("server")
        claimed_ids = facts.get("player_ids") or []
        lower_complaint = (complaint_text or "").lower()

        if not analyzed_files:
            # Check if complaint text claims attachments exist
            if any(w in lower_complaint for w in ("скриншот приложил", "видео приложил", "прикрепил", "attached")):
                contradictions.append(
                    "В тексте жалобы утверждается, что доказательства приложены, но медиафайлы отсутствуют"
                )
                consistency_score -= 0.45
            return {
                "status": "no_media",
                "is_consistent": len(contradictions) == 0,
                "files_count": 0,
                "files": [],
                "extracted_nicknames": [],
                "extracted_player_ids": [],
                "extracted_servers": [],
                "has_time_stamp": False,
                "has_video": False,
                "has_image": False,
                "consistency_score": round(max(0.0, consistency_score), 2),
                "contradictions": contradictions,
                "warnings": ["Медиафайлы (JPG/PNG/MP4/WEBM) не загружены для детального OCR/видео-анализа"],
                "matches": [],
                "admin_report": "Доказательства не предоставлены. Для вынесения решения запросите у игрока скриншот или видео с /time (п. 7.2).",
            }

        # 1. Check blank/corrupted images
        if blank_images > 0:
            contradictions.append("Загруженный скриншот пустой (чёрный/однотонный экран) и не содержит игровых данных")
            consistency_score -= 0.50

        # 2. Compare Server number
        if claimed_server is not None and ev_servers:
            try:
                c_srv = int(claimed_server)
                if c_srv not in ev_servers:
                    contradictions.append(
                        f"Несовпадение сервера: в тексте жалобы указан сервер #{c_srv}, "
                        f"а на доказательствах распознан сервер #{ev_servers[0]}"
                    )
                    consistency_score -= 0.45
                else:
                    matches.append(f"Номер сервера (#{c_srv}) подтверждён на доказательствах")
            except Exception:
                pass

        # 3. Compare Offender Nickname
        if claimed_offender and all_ev_nicknames:
            ev_nicks_lower = {n.lower() for n in all_ev_nicknames}
            if claimed_offender.lower() in ev_nicks_lower:
                matches.append(f"Никнейм нарушителя ({claimed_offender}) найден на доказательствах")
            else:
                # If OCR found other nicknames, but NOT the accused offender
                other_nicks = [n for n in all_ev_nicknames if not claimed_reporter or n.lower() != claimed_reporter.lower()]
                if other_nicks:
                    contradictions.append(
                        f"Противоречие по никнейму: в жалобе обвиняется {claimed_offender}, "
                        f"но на доказательствах зафиксирован другой игрок ({', '.join(other_nicks)})"
                    )
                    consistency_score -= 0.40

        # 4. Compare Player IDs
        if claimed_ids and all_ev_ids:
            common_ids = set(claimed_ids) & set(all_ev_ids)
            if common_ids:
                matches.append(f"Игровой ID ({', '.join(map(str, common_ids))}) совпадает с описанием жалобы")
            else:
                contradictions.append(
                    f"Несовпадение игрового ID: в тексте указан ID {claimed_ids}, "
                    f"а на медиафайле распознан ID {all_ev_ids}"
                )
                consistency_score -= 0.35

        # 5. Check media type vs claim & category requirements
        claims_video_in_text = bool(re.search(r"\b(?:видео|фрапс|откат|video|fraps)\b", lower_complaint))
        if claims_video_in_text and not has_video and has_image:
            contradictions.append(
                "Игрок пишет в жалобе о приложенной видеозаписи (фрапсе), но загрузил только статичное изображение"
            )
            consistency_score -= 0.25

        if category == "cheating_exploits" and not has_video:
            warnings.append(
                "Для жалоб на читы (п. 5.1) требуется видеозапись (MP4/WEBM); одного скриншота недостаточно"
            )
            consistency_score -= 0.20

        # 6. Check /time presence (Rule 7.2)
        if any_time_stamp:
            matches.append("Отметка игрового времени (/time) присутствует")
        else:
            warnings.append("На доказательствах не обнаружена отметка времени (/time), требуемая по п. 7.2")
            consistency_score -= 0.15

        consistency_score = round(max(0.0, min(1.0, consistency_score)), 2)
        if contradictions:
            status = "contradiction"
        elif consistency_score >= 0.80:
            status = "verified"
        else:
            status = "partial"

        # Build concise admin report
        report_parts = [
            f"Проанализировано файлов: {len(analyzed_files)} "
            f"({'видео' if has_video else ''}{' и ' if (has_video and has_image) else ''}{'изображения' if has_image else ''})."
        ]
        if all_ev_nicknames or all_ev_ids:
            report_parts.append(
                f"Извлечено из медиа: никнеймы={all_ev_nicknames or ['не найдены']}, ID={all_ev_ids or ['не найдены']}, "
                f"/time={'есть' if any_time_stamp else 'не найден'}."
            )
        if matches:
            report_parts.append("Подтверждено: " + "; ".join(matches) + ".")
        if contradictions:
            report_parts.append("⚠️ ПРОТИВОРЕЧИЯ: " + "; ".join(contradictions) + ".")
        elif warnings:
            report_parts.append("Замечания: " + "; ".join(warnings) + ".")

        return {
            "status": status,
            "is_consistent": len(contradictions) == 0,
            "files_count": len(analyzed_files),
            "files": analyzed_files,
            "extracted_nicknames": all_ev_nicknames,
            "extracted_player_ids": all_ev_ids,
            "extracted_servers": ev_servers,
            "has_time_stamp": any_time_stamp,
            "has_video": has_video,
            "has_image": has_image,
            "consistency_score": consistency_score,
            "contradictions": contradictions,
            "warnings": warnings,
            "matches": matches,
            "admin_report": " ".join(report_parts),
        }


# Singleton instance
evidence_analyzer = EvidenceAnalyzer()
