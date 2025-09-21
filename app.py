from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
import tempfile, datetime

app = FastAPI()

@app.post("/fixdicom")
async def fix_dicom(file: UploadFile, rows: int, cols: int, bits: int = 16):
    raw = await file.read()

    # --- File Meta Info ---
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # --- Create Part 10 DICOM ---
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

    # Add acquisition date/time
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".dcm")
    ds.save_as(tmp.name, write_like_original=False)

    return FileResponse(tmp.name, media_type="application/dicom", filename="fixed.dcm")
