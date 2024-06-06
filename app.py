from fastapi import FastAPI
from fastapi.responses import JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
import cv2
import numpy as np
import openai
import torch
from dotenv import load_dotenv
import os
from doctr.file_utils import is_tf_available
from doctr.utils.visualization import visualize_page
from doctr.models import ocr_predictor
import pandas as pd
from backend.pytorch import DET_ARCHS, RECO_ARCHS, forward_image, load_predictor
from pymongo import MongoClient
import base64
from PIL import Image
import io
from datetime import datetime

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
mongo_url = os.getenv("MONGODB_API_KEY")

# Set up device for PyTorch
forward_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Connect to MongoDB
client = MongoClient(mongo_url)
db = client["Documenttask"]
images_collection = db["images"]

# Predefined questions array
questions = [
    "Title Of The Document", "Instrument Number", "Book", "Page", "Recorded Date",
    "Buyer/ Borrower", "Seller/ Lender",
    "Full Legal Description", "Parcel ID/ APN", "Situs/ Property Address"
]

# Function to extract text data from OCR results
def extract_text_data(ocr_result):
    pages_text = []
    for page in ocr_result.pages:
        page_data = {"page_idx": page.page_idx, "text": ""}
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    page_data["text"] += word.value + " "
                page_data["text"] += "\n"
        pages_text.append(page_data)
    return pages_text

# Function to get response from OpenAI
def get_openai_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Always provide concise answers without restating the question. Date as YYYYMMDD format. Names as Last Name, First Name Middle Name format."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500
    )
    return response.choices[0].message['content'].strip()

# Function to decode and load images from a multi-page TIFF Base64 string
def load_images_from_base64(base64_data):
    image_data = base64.b64decode(base64_data)
    images = []
    
    with Image.open(io.BytesIO(image_data)) as img:
        for i in range(img.n_frames):
            img.seek(i)
            img_page = img.convert("RGB")
            img_page = np.array(img_page)
            img_page = cv2.cvtColor(img_page, cv2.COLOR_RGB2BGR)
            images.append(img_page)
    
    return images

# Function to encode image to base64
def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

# Main function to process documents
def process_documents():
    print("Checking for new documents...")
    predictor = load_predictor(
        det_arch=DET_ARCHS[0],  # Select the default or desired model
        reco_arch=RECO_ARCHS[0],  # Select the default or desired model
        assume_straight_pages=True,
        straighten_pages=False,
        bin_thresh=0.3,
        box_thresh=0.1,
        device=forward_device
    )

    # Process documents with status "processing" first
    documents_processing = list(images_collection.find({"status": "processing"}))
    documents_notprocessed = list(images_collection.find({"status": "notprocessed"}))

    if not documents_processing and not documents_notprocessed:
        print("All documents are processed.")
        return

    def process_document(document):
        # Update document status to "processing"
        images_collection.update_one({"_id": document["_id"]}, {"$set": {"status": "processing"}})

        base64_image = document["image"]
        try:
            images = load_images_from_base64(base64_image)
            doc = images  # List of images (pages)
        except ValueError as e:
            print(f"Failed to load images: {e}")
            return

        print(f"Processing document: {document['filename']}")

        all_pages_text = []  # Store text data from all pages
        input_pages = []  # Store input pages
        segmentation_heatmaps = []  # Store segmentation heatmaps
        ocr_outputs = []  # Store OCR outputs
        page_reconstitutions = []  # Store page reconstitutions

        for page_idx, page in enumerate(doc):
            print(f"Processing page {page_idx + 1}/{len(doc)}...")
            # Forward image through the model
            seg_map = forward_image(predictor, page, forward_device)
            seg_map = np.squeeze(seg_map)
            seg_map = cv2.resize(seg_map, (page.shape[1], page.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Perform OCR
            out = predictor([page])
            
            # Extract text data
            pages_text = extract_text_data(out)
            all_pages_text.extend([page["text"] for page in pages_text])  # Extract and append text data correctly

            # Encode images to base64
            input_pages.append(encode_image_to_base64(page))
            segmentation_heatmaps.append(encode_image_to_base64(seg_map))
            ocr_outputs.append(encode_image_to_base64(out.pages[0].synthesize()))
            page_reconstitutions.append(encode_image_to_base64(out.pages[0].synthesize()))

        # Store extracted text in session state
        document_text = "\n".join(all_pages_text)

        # Get the current date
        current_date = datetime.now().strftime("%Y-%m-%d")

        # Handle predefined questions and responses
        prompts_questions = questions
        prompts_results = []

        for question in questions:
            prompt = f"Document text:\n{document_text}\n\nQuestion: {question}\n\nAnswer (provide a precise and concise response without restating the question):"
            answer = get_openai_response(prompt)
            prompts_results.append(answer)

        # Update the document in MongoDB
        images_collection.update_one({"_id": document["_id"]}, {"$set": {
            "status": "processed",
            "processed_date": current_date,
            "json_data": all_pages_text,
            "prompts_questions": prompts_questions,
            "prompts_results": prompts_results,
            "input_page": input_pages,
            # "segmentation_heatmap": segmentation_heatmaps,
            # "ocr_output": ocr_outputs,
            "page_reconstitution": page_reconstitutions
        }})
        
        print(f"Document {document['filename']} processed and updated in MongoDB.")

    # Process documents that were previously in "processing" state
    for document in documents_processing:
        process_document(document)

    # Process documents that are "notprocessed"
    for document in documents_notprocessed:
        process_document(document)

# Initialize FastAPI app
app = FastAPI()

# Schedule the process_documents function to run on an interval
scheduler = BackgroundScheduler()
scheduler.add_job(process_documents, 'interval', minutes=10)
scheduler.start()

@app.post("/process_documents")
async def process_documents_api():
    process_documents()
    return JSONResponse(content={"message": "Documents processed successfully"}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
