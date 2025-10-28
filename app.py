from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.errors import InvalidDicomError
import tempfile, datetime, os, zipfile, io, traceback, pydicom

app = FastAPI(title="DICOM Fixer API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dev.apollo-test.com",
        "https://app.apollo-test.com",
        "http://localhost:5173",
        "https://dicom-fixer.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fix_single_dicom(ds: FileDataset) -> bytes:
    """Ensure a DICOM dataset has required metadata and return fixed file bytes."""
    print(f"[DEBUG] Fixing DICOM {getattr(ds, 'PatientID', 'Unknown ID')} ...")

    # Ensure minimal required metadata
    ds.SOPClassUID = getattr(ds, "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2")
    ds.SOPInstanceUID = getattr(ds, "SOPInstanceUID", generate_uid())
    ds.Modality = getattr(ds, "Modality", "CT")
    ds.PatientName = getattr(ds, "PatientName", "Anonymous")
    ds.PatientID = getattr(ds, "PatientID", "12345")
    ds.StudyInstanceUID = getattr(ds, "StudyInstanceUID", generate_uid())
    ds.SeriesInstanceUID = getattr(ds, "SeriesInstanceUID", generate_uid())
    ds.ImagePositionPatient = getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0])
    ds.ImageOrientationPatient = getattr(
        ds, "ImageOrientationPatient", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    ds.PixelSpacing = getattr(ds, "PixelSpacing", [1.0, 1.0])
    ds.SliceThickness = getattr(ds, "SliceThickness", 1.0)
    ds.PhotometricInterpretation = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")

    # Add content date/time if missing
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S")

    # Save to bytes
    with io.BytesIO() as buffer:
        ds.save_as(buffer, write_like_original=False)
        print("[DEBUG] DICOM saved successfully")
        return buffer.getvalue()


@app.post("/fixdicom-zip")
async def fix_dicom_zip(file: UploadFile):
    """Accept a ZIP file of DICOM files, fix each one, and return a ZIP of corrected DICOMs."""
    print(f"[INFO] Received ZIP: {file.filename}")

    try:
        zip_bytes = await file.read()
        print(f"[DEBUG] ZIP size: {len(zip_bytes)} bytes")

        with tempfile.TemporaryDirectory() as tmpdir:
            input_zip_path = os.path.join(tmpdir, "input.zip")
            with open(input_zip_path, "wb") as f:
                f.write(zip_bytes)

            # Extract uploaded ZIP
            with zipfile.ZipFile(input_zip_path, "r") as zip_ref:
                zip_ref.extractall(tmpdir)
            extracted = os.listdir(tmpdir)
            print(f"[DEBUG] Extracted files: {extracted}")

            total_files = 0
            error_files = 0

            # Prepare output ZIP in memory
            output_zip_bytes = io.BytesIO()
            with zipfile.ZipFile(output_zip_bytes, "w", zipfile.ZIP_DEFLATED) as output_zip:
                for name in extracted:
                    if not name.lower().endswith((".dcm", ".raw", ".bin")):
                        print(f"[DEBUG] Skipping non-DICOM file: {name}")
                        continue

                    total_files += 1
                    input_path = os.path.join(tmpdir, name)
                    print(f"[INFO] Processing {name} ...")

                    try:
                        # Try reading as DICOM
                        ds = pydicom.dcmread(input_path, force=True)
                        print(f"[DEBUG] Loaded DICOM: Rows={getattr(ds, 'Rows', 'N/A')}, "
                              f"Cols={getattr(ds, 'Columns', 'N/A')}, Bits={getattr(ds, 'BitsAllocated', 'N/A')}")

                        fixed_data = fix_single_dicom(ds)
                        output_zip.writestr(f"fixed_{name}", fixed_data)
                        print(f"[INFO] Successfully fixed {name}")

                    except InvalidDicomError:
                        error_files += 1
                        msg = f"{name} is not a valid DICOM file (missing header)"
                        print("[ERROR]", msg)
                        output_zip.writestr(f"error_{name}.txt", msg)

                    except Exception as e:
                        error_files += 1
                        msg = f"Error fixing {name}: {str(e)}\n{traceback.format_exc()}"
                        print("[ERROR]", msg)
                        output_zip.writestr(f"error_{name}.txt", msg)

            print(f"[SUMMARY] Processed {total_files} files, {error_files} errors")

            output_zip_bytes.seek(0)
            return StreamingResponse(
                output_zip_bytes,
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="fixed_dicoms.zip"'
                },
            )

    except zipfile.BadZipFile:
        print("[ERROR] Uploaded file is not a valid ZIP archive")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid ZIP archive.")
    except Exception as e:
        print("[FATAL] Unhandled exception in fix_dicom_zip:")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
