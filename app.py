from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import tempfile, datetime, os, zipfile, io

app = FastAPI(title="DICOM Fixer API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fix_single_dicom(raw: bytes, rows: int, cols: int, bits: int = 16) -> bytes:
    """Creates a valid DICOM file from raw pixel bytes and returns the file content as bytes."""
    expected_len = rows * cols * (bits // 8)
    if len(raw) != expected_len:
        raise ValueError(f"Pixel data size mismatch: expected {expected_len}, got {len(raw)}")

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "CT"
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated = bits
    ds.BitsStored = bits
    ds.HighBit = bits - 1
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = raw

    # Add some metadata
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S")
    ds.PatientName = "Anonymous"
    ds.PatientID = "12345"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.ImagePositionPatient = [0.0, 0.0, 0.0]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 1.0

    with io.BytesIO() as buffer:
        ds.save_as(buffer, write_like_original=False)
        return buffer.getvalue()


@app.post("/fixdicom-zip")
async def fix_dicom_zip(file: UploadFile, rows: int, cols: int, bits: int = 16):
    """
    Accept a ZIP file of DICOM (or raw) files, fix each one, and return a ZIP of corrected DICOMs.
    """
    try:
        zip_bytes = await file.read()
        with tempfile.TemporaryDirectory() as tmpdir:
            input_zip_path = os.path.join(tmpdir, "input.zip")
            with open(input_zip_path, "wb") as f:
                f.write(zip_bytes)

            # Extract uploaded ZIP
            with zipfile.ZipFile(input_zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)

            # Prepare output ZIP in memory
            output_zip_bytes = io.BytesIO()
            with zipfile.ZipFile(output_zip_bytes, "w", zipfile.ZIP_DEFLATED) as output_zip:
                for name in os.listdir(tmpdir):
                    if not name.lower().endswith((".dcm", ".raw", ".bin")):
                        continue
                    input_path = os.path.join(tmpdir, name)
                    with open(input_path, "rb") as f:
                        raw = f.read()

                    try:
                        fixed_data = fix_single_dicom(raw, rows, cols, bits)
                        output_zip.writestr(f"fixed_{name}", fixed_data)
                    except Exception as e:
                        # Include a small error file instead of skipping silently
                        output_zip.writestr(f"error_{name}.txt", str(e))

            output_zip_bytes.seek(0)
            tmp_output_path = os.path.join(tmpdir, "fixed_dicom_files.zip")
            with open(tmp_output_path, "wb") as f:
                f.write(output_zip_bytes.read())

            return FileResponse(
                tmp_output_path,
                media_type="application/zip",
                filename="fixed_dicoms.zip",
            )

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
