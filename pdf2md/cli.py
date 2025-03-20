from mistralai import Mistral
from pathlib import Path
import os
import base64
import sys
import urllib.parse
from mistralai import DocumentURLChunk
from mistralai.models import OCRResponse

def replace_images_in_markdown(markdown_str: str, images_dict: dict) -> str:
    for img_name, img_path in images_dict.items():
        markdown_str = markdown_str.replace(f"![{img_name}]({img_name})", f"![{img_name}]({img_path})")
    return markdown_str

def process_pdf_to_md(pdf_path: str, api_key: str) -> str:
    # Initialize client
    client = Mistral(api_key=api_key)
    
    # Confirm PDF file exists
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    
    # Create output directory for images
    pdf_name = pdf_file.stem
    pdf_parent_dir = pdf_file.parent
    images_dir = pdf_parent_dir / f"{pdf_name}_images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Upload and process PDF
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_name,
            "content": pdf_file.read_bytes(),
        },
        purpose="ocr",
    )
    
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url), 
        model="mistral-ocr-latest",
        include_image_base64=True
    )
    
    # Process pages and images
    all_markdowns = []
    for page in pdf_response.pages:
        # Save images
        page_images = {}
        for img in page.images:
            img_data = base64.b64decode(img.image_base64.split(',')[1])
            img_path = images_dir / f"{img.id}.png"
            with open(img_path, 'wb') as f:
                f.write(img_data)
            # Use relative path instead of absolute path
            relative_img_path = f"{pdf_name}_images/{img.id}.png"
            # URL encode spaces and special characters for markdown
            encoded_img_path = urllib.parse.quote(relative_img_path).replace(' ', '%20')
            page_images[img.id] = encoded_img_path
        
        # Process markdown content
        page_markdown = replace_images_in_markdown(page.markdown, page_images)
        all_markdowns.append(page_markdown)
    
    # Join all pages' markdown content
    return "\n\n".join(all_markdowns)

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: pdf2md name.pdf")
        sys.exit(1)
    
    # Get PDF path from command line
    pdf_path = sys.argv[1]
    
    # Get API key from environment variable
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set")
        print("Please set it with: export MISTRAL_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Process PDF and get markdown content
    try:
        md_content = process_pdf_to_md(pdf_path, api_key)
        
        # Create output MD file with the same name as the input PDF
        output_md_path = Path(pdf_path).with_suffix('.md')
        
        # Write markdown content to output file
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Conversion complete. Result saved to: {output_md_path}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()