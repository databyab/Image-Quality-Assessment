from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
import csv
import json
import re
import sys
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app.inference import LABEL_COLUMNS, PhotoQualityPredictor

MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "baseline"
    / "checkpoints"
    / "best_model.pt"
)

LABELS = {
    "blur": "Blur",
    "noise": "Noise",
    "underexposure": "Underexposure",
    "overexposure": "Overexposure",
    "compression": "Compression",
}
NO_DEFECT_LABEL = "No detected defects"
MAX_SINGLE_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_MULTIPLE_UPLOAD_FILE_BYTES = 10 * 1024 * 1024
MAX_MULTIPLE_UPLOAD_TOTAL_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


def clear_classification():
    st.session_state["classification"] = None


@st.cache_resource
def load_model():
    return PhotoQualityPredictor(str(MODEL_PATH))


def read_uploaded_image(uploaded_file, max_upload_bytes):
    raw_data = uploaded_file.getvalue()
    if len(raw_data) > max_upload_bytes:
        max_size_mb = max_upload_bytes // (1024 * 1024)
        raise ValueError(
            f"Image exceeds this mode's {max_size_mb} MB per-file limit."
        )

    try:
        with Image.open(BytesIO(raw_data)) as decoded_image:
            width, height = decoded_image.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("Image dimensions are too large.")
            decoded_image.load()
            return decoded_image.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("This file is not a valid image.") from exc


def analyze_image(image, uploaded_file, predictor):
    suffix = ".png" if uploaded_file.type == "image/png" else ".jpg"
    image_format = "PNG" if suffix == ".png" else "JPEG"
    temporary_file = tempfile.NamedTemporaryFile(
        prefix="photoiq_",
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(temporary_file.name)
    try:
        temporary_file.close()
        image.save(temp_path, format=image_format)
        return predictor.predict(str(temp_path))
    finally:
        temp_path.unlink(missing_ok=True)


def safe_relative_name(filename):
    normalized = filename.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part not in ("", ".", "..")]
    safe_parts = [
        re.sub(r'[<>:"|?*\x00-\x1f]', "_", part)
        for part in parts
    ]
    return "/".join(safe_parts) if safe_parts else "photo"


def unique_zip_path(existing_paths, candidate):
    path = PurePosixPath(candidate)
    parent = "" if str(path.parent) == "." else f"{path.parent}/"
    stem, suffix = path.stem, path.suffix
    result = candidate
    counter = 2
    while result in existing_paths:
        result = f"{parent}{stem} ({counter}){suffix}"
        counter += 1
    existing_paths.add(result)
    return result


def build_archive(uploaded_files, records):
    buffer = BytesIO()
    labels = list(LABELS.values()) + [NO_DEFECT_LABEL]
    used_paths = {label: set() for label in labels}

    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for label in labels:
            archive.writestr(f"{label}/", "")

        for record in records:
            uploaded_file = uploaded_files[record["file_index"]]
            relative_name = safe_relative_name(uploaded_file.name)
            for label in record["labels"]:
                entry = unique_zip_path(
                    used_paths[label],
                    f"{label}/{relative_name}",
                )
                archive.writestr(entry, uploaded_file.getvalue())

    return buffer.getvalue()


def build_manifest(records, thresholds, errors):
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["source", "labels", "status", "probabilities", "thresholds"],
    )
    writer.writeheader()

    for record in records:
        writer.writerow({
            "source": record["filename"],
            "labels": "; ".join(record["labels"]),
            "status": "classified",
            "probabilities": json.dumps(record["probabilities"], sort_keys=True),
            "thresholds": json.dumps(thresholds, sort_keys=True),
        })
    for error in errors:
        writer.writerow({
            "source": error["filename"],
            "labels": "",
            "status": error["error"],
            "probabilities": "{}",
            "thresholds": json.dumps(thresholds, sort_keys=True),
        })
    return buffer.getvalue()


def show_report(uploaded_files, classification):
    records = classification["records"]
    errors = classification["errors"]
    counts = {label: 0 for label in list(LABELS.values()) + [NO_DEFECT_LABEL]}

    for record in records:
        for label in record["labels"]:
            counts[label] += 1

    st.subheader("Classification report")
    st.caption(
        f"{len(records)} photos classified; {len(errors)} file(s) skipped. "
        "Photos with multiple detected defects appear in each matching label count."
    )
    st.dataframe(
        [{"Label": label, "Photos": count} for label, count in counts.items()],
        hide_index=True,
        use_container_width=True,
    )

    download_col, report_col = st.columns(2)
    with download_col:
        st.download_button(
            "Download labeled folders (ZIP)",
            data=classification["archive"],
            file_name="photoiq_classified_photos.zip",
            mime="application/zip",
            use_container_width=True,
        )
    with report_col:
        st.download_button(
            "Download classification report (CSV)",
            data=build_manifest(
                records,
                classification["thresholds"],
                errors,
            ),
            file_name="photoiq_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    available_labels = list(counts)
    selected_label = st.selectbox("Open label folder", available_labels)
    st.markdown(f"#### {selected_label} ({counts[selected_label]} photos)")
    selected_records = [
        record for record in records if selected_label in record["labels"]
    ]

    if not selected_records:
        st.info("This label folder is empty for the current selection.")
    else:
        for offset in range(0, len(selected_records), 3):
            columns = st.columns(3)
            for column, record in zip(columns, selected_records[offset:offset + 3]):
                with column:
                    uploaded_file = uploaded_files[record["file_index"]]
                    st.image(
                        uploaded_file.getvalue(),
                        caption=record["filename"],
                        use_container_width=True,
                    )
                    if selected_label == NO_DEFECT_LABEL:
                        st.caption("No defect score met its configured threshold.")
                    else:
                        st.caption(
                            f"{selected_label}: "
                            f"{record['probabilities'].get(selected_label, 0.0):.1%} "
                            f"(threshold "
                            f"{classification['thresholds'].get(selected_label, 0.5):.0%})"
                        )

    if errors:
        with st.expander("Skipped files"):
            for error in errors:
                st.write(f"{error['filename']}: {error['error']}")


st.set_page_config(
    page_title="PhotoIQ",
    page_icon="\U0001F4F7",
    layout="wide",
)

st.title("\U0001F4F7 PhotoIQ")
st.subheader("Photo quality classification")
st.write(
    "Upload a photo, a batch, or a folder. PhotoIQ will classify technical "
    "defects, report counts, and let you browse each label."
)

upload_mode = st.radio(
    "Upload mode",
    ["Single photo", "Multiple photos", "Folder"],
    horizontal=True,
    key="upload_mode",
    on_change=clear_classification,
)

if upload_mode == "Single photo":
    accept_multiple_files = False
    uploader_label = "Choose a photo"
    per_file_limit = MAX_SINGLE_UPLOAD_BYTES
elif upload_mode == "Folder":
    accept_multiple_files = "directory"
    uploader_label = "Choose a folder"
    per_file_limit = MAX_MULTIPLE_UPLOAD_FILE_BYTES
else:
    accept_multiple_files = True
    uploader_label = "Choose photos"
    per_file_limit = MAX_MULTIPLE_UPLOAD_FILE_BYTES

uploaded_files = st.file_uploader(
    uploader_label,
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=accept_multiple_files,
    max_upload_size=per_file_limit // (1024 * 1024),
    key="photo_uploader",
    on_change=clear_classification,
)

st.write("### Detection thresholds")
thresholds = {}
threshold_columns = st.columns(5)
for column, label in zip(threshold_columns, LABEL_COLUMNS):
    with column:
        thresholds[LABELS[label]] = st.slider(
            LABELS[label],
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key=f"threshold_{label}",
            on_change=clear_classification,
            help="A label is assigned when its model score meets or exceeds this value.",
        )

if uploaded_files and not isinstance(uploaded_files, list):
    uploaded_files = [uploaded_files]

if uploaded_files:
    if upload_mode == "Single photo":
        total_limit = MAX_SINGLE_UPLOAD_BYTES
    else:
        total_limit = MAX_MULTIPLE_UPLOAD_TOTAL_BYTES

    total_bytes = sum(len(file.getvalue()) for file in uploaded_files)
    total_limit_mb = total_limit // (1024 * 1024)
    per_file_limit_mb = per_file_limit // (1024 * 1024)
    st.caption(
        f"Per-photo limit: {per_file_limit_mb} MB. "
        + (
            f"Total batch limit: {total_limit_mb} MB."
            if upload_mode != "Single photo"
            else f"Maximum file size: {total_limit_mb} MB."
        )
    )

    within_batch_limit = total_bytes <= total_limit
    if not within_batch_limit:
        st.error(
            f"This selection is {total_bytes / (1024 * 1024):.1f} MB. "
            f"The limit for this mode is {total_limit_mb} MB total."
        )

    if st.button(
        "Classify photos",
        type="primary",
        disabled=not within_batch_limit,
    ):
        predictor = load_model()
        records = []
        errors = []
        progress = st.progress(0, text="Preparing photos")

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                image = read_uploaded_image(uploaded_file, per_file_limit)
                raw_results = analyze_image(image, uploaded_file, predictor)
            except (ValueError, OSError, RuntimeError) as exc:
                errors.append({
                    "filename": safe_relative_name(uploaded_file.name),
                    "error": str(exc),
                })
                progress.progress(
                    (index + 1) / len(uploaded_files),
                    text=f"Skipped {uploaded_file.name}",
                )
                continue

            probabilities = {
                LABELS[label]: result["probability"]
                for label, result in raw_results.items()
            }
            detected_labels = [
                display_label
                for display_label, probability in probabilities.items()
                if probability >= thresholds[display_label]
            ]
            record_labels = detected_labels or [NO_DEFECT_LABEL]
            records.append({
                "file_index": index,
                "filename": safe_relative_name(uploaded_file.name),
                "labels": record_labels,
                "probabilities": probabilities,
            })
            progress.progress(
                (index + 1) / len(uploaded_files),
                text=f"Classified {uploaded_file.name}",
            )

        progress.empty()
        st.session_state["classification"] = {
            "records": records,
            "errors": errors,
            "thresholds": thresholds,
            "archive": build_archive(uploaded_files, records),
        }

classification = st.session_state.get("classification")
if classification is not None and uploaded_files:
    show_report(uploaded_files, classification)
elif classification is not None:
    clear_classification()
