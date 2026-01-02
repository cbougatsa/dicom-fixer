from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
from pydicom.errors import InvalidDicomError
import tempfile, datetime, os, zipfile, io, traceback, pydicom
import nibabel as nib
import numpy as np

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


@app.post("/dicom/fix")
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


@app.post("/dicom/nifti-convert")
async def nifti_to_dicom_zip(file: UploadFile):
    """
    Convert a .nii or .nii.gz file to a ZIP of DICOM slices
    """
    print(f"[INFO] Received NIfTI: {file.filename}")

    if not file.filename.lower().endswith((".nii", ".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii or .nii.gz files are supported")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            nifti_path = os.path.join(tmpdir, file.filename)
            with open(nifti_path, "wb") as f:
                f.write(await file.read())

            # Load NIfTI
            nii = nib.load(nifti_path)
            data = nii.get_fdata()
            affine = nii.affine

            # Normalize to int16 (DICOM-friendly)
            data = np.asarray(data, dtype=np.int16)

            # Volume shape
            rows, cols, slices = data.shape
            print(f"[DEBUG] Volume shape: {data.shape}")

            # Shared UIDs
            study_uid = generate_uid()
            series_uid = generate_uid()

            output_zip = io.BytesIO()
            with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zip_out:
                for i in range(slices):
                    ds = FileDataset(
                        None,
                        {},
                        file_meta=Dataset(),
                        preamble=b"\0" * 128,
                    )

                    # File meta
                    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
                    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()

                    # Core identifiers
                    ds.SOPClassUID = CTImageStorage
                    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
                    ds.StudyInstanceUID = study_uid
                    ds.SeriesInstanceUID = series_uid
                    ds.Modality = "CT"

                    # Patient (dummy / anonymized)
                    ds.PatientName = "Anonymous"
                    ds.PatientID = "NIFTI001"

                    # Image geometry
                    ds.Rows = rows
                    ds.Columns = cols
                    ds.InstanceNumber = i + 1
                    ds.ImagePositionPatient = [
                        float(affine[0, 3]),
                        float(affine[1, 3]),
                        float(affine[2, 3] + i),
                    ]
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                    ds.PixelSpacing = [1.0, 1.0]
                    ds.SliceThickness = 1.0

                    # Pixel data
                    slice_data = data[:, :, i]
                    ds.PixelData = slice_data.tobytes()
                    ds.BitsAllocated = 16
                    ds.BitsStored = 16
                    ds.HighBit = 15
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = "MONOCHROME2"
                    ds.PixelRepresentation = 1

                    # Dates
                    now = datetime.datetime.now()
                    ds.ContentDate = now.strftime("%Y%m%d")
                    ds.ContentTime = now.strftime("%H%M%S")

                    # Write slice
                    with io.BytesIO() as buffer:
                        ds.save_as(buffer, write_like_original=False)
                        zip_out.writestr(f"{i:04d}.dcm", buffer.getvalue())

            output_zip.seek(0)
            return StreamingResponse(
                output_zip,
                media_type="application/zip",
                headers={
                    "Content-Disposition": 'attachment; filename="dicom_from_nifti.zip"'
                },
            )

    except Exception as e:
        print("[ERROR] NIfTI conversion failed")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
