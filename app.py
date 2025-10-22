from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import tempfile, datetime, os

app = FastAPI(title="DICOM Fixer API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your domain, e.g. ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/fixdicom")
async def fix_dicom(file: UploadFile, rows: int, cols: int, bits: int = 16):
    try:
        raw = await file.read()
        expected_len = rows * cols * (bits // 8)
        if len(raw) != expected_len:
            raise HTTPException(
                status_code=400,
                detail=f"Pixel data size mismatch: expected {expected_len} bytes, got {len(raw)} bytes.",
            )

        # --- File Meta Info ---
        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        # --- Create DICOM Dataset ---
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

        # --- Add metadata ---
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

        # --- Write to temp file ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
            ds.save_as(tmp.name, write_like_original=False)
            tmp_path = tmp.name

        # --- Return as FileResponse and clean up after sending ---
        response = FileResponse(
            tmp_path,
            media_type="application/dicom",
            filename="fixed.dcm",
        )

        # Schedule deletion after response is sent
        @response.call_on_close
        def cleanup_temp():
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return response

    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
